"""현재 대피 계산을 계측하고, 분리된 Qwen 요청 표본을 실제 추론한다.

운영 소스·결과·캐시를 변경하지 않는다. 재현 실행 중 캐시 미스는 즉시 중단한다.
중첩 구간은 제외 시간으로 합산하고, Qwen 병렬 요청 시간은 주 스레드 대기와 구분한다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import cProfile
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import threading
import time


class PhaseTimer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.main_thread = threading.get_ident()
        self.local = threading.local()
        self.rows = defaultdict(lambda: {"calls": 0, "inclusive_s": 0., "exclusive_s": 0.})

    @contextmanager
    def span(self, name, category):
        # Worker requests run inside a main-thread batch wait. Do not double count them.
        if threading.get_ident() != self.main_thread:
            yield
            return
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        frame = {"start": self.clock(), "children": 0.}
        stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - frame["start"]
            stack.pop()
            row = self.rows[(category, name)]
            row["calls"] += 1
            row["inclusive_s"] += elapsed
            row["exclusive_s"] += max(0., elapsed - frame["children"])
            if stack:
                stack[-1]["children"] += elapsed

    def wrap(self, owner, name, category):
        original = getattr(owner, name)
        label = f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.span(label, category):
                return original(*args, **kwargs)

        setattr(owner, name, measured)

    def report(self):
        totals = defaultdict(float)
        rows = []
        for (category, name), values in self.rows.items():
            totals[category] += values["exclusive_s"]
            rows.append({"category": category, "function": name, **values})
        total = sum(totals.values())
        return {
            "total_s": total,
            "categories": [
                {"category": k, "seconds": v, "percent": 100*v/total if total else 0}
                for k, v in sorted(totals.items(), key=lambda p: -p[1])
            ],
            "functions": sorted(rows, key=lambda r: -r["exclusive_s"]),
        }


class CacheMiss(BaseException):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    from persona_evac.llm.ollama import atomic_json
    atomic_json(path, value)


def model_state(local):
    try:
        response = local.client.get("/api/ps")
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"unavailable": True, "error": str(exc)}


def qwen_summary(records, groups, workers):
    valid = [r for r in records if r["status"] == "valid"]
    offset, valid_groups = 0, []
    for group in groups:
        group_records = records[offset:offset+group["count"]]
        if len(group_records) == group["count"] and all(r["status"] == "valid" for r in group_records):
            valid_groups.append(group)
        offset += group["count"]
    return {"count": len(records), "valid_count": len(valid),
        "failed_count": len(records)-len(valid), "workers": workers,
        "batch_wall_total_s": sum(g["wall_s"] for g in groups),
        "valid_batch_wall_total_s": sum(g["wall_s"] for g in valid_groups),
        "request_wall_sum_s": sum(r["wall_s"] for r in valid),
        "request_median_s": statistics.median(r["wall_s"] for r in valid) if valid else None,
        "request_min_s": min((r["wall_s"] for r in valid), default=None),
        "request_max_s": max((r["wall_s"] for r in valid), default=None),
        "scope": "실제 요청 표본이며 운영 결과·캐시에 적용하지 않는다. 지연 통계에서는 실패 요청을 제외한다."}


def finalize_existing(out):
    """서버 재시작으로 중단된 보고서를 저장된 표본과 출처 검증으로 마무리한다."""
    from persona_evac.integration.station import source_manifest
    from persona_evac.integration.fds_environment import FireField
    from persona_evac.station_artifacts import station_signature
    report = json.loads((out/"report.json").read_bytes())
    samples = json.loads((out/"qwen_samples.json").read_bytes())
    baseline = json.loads(Path("output/gaehwasan/results.json").read_bytes())
    source_path = Path("output/gaehwasan/persona_source.json")
    source = json.loads(source_path.read_bytes())
    originals = sorted(source["variants"]["C"], key=lambda a: a["agent_slot"])[:report["configuration"]["agents"]]
    expected_source = source_manifest(source, originals)
    baseline_source = dict(baseline["source"])
    attached_hash = baseline_source.pop("artifact_sha256")
    report["source_metadata_validation"] = {
        "manifest_matches": expected_source == baseline_source,
        "input_hash_matches": hashlib.sha256(source_path.read_bytes()).hexdigest() == attached_hash,
        "note": "계측은 simulate_station을 직접 호출했다. 실행 래퍼가 나중에 붙이는 artifact_sha256은 원본 파일 해시로 별도 검증했다."}
    report["code_unchanged"] = station_signature() == report["implementation_sha256"]
    # FireField checks the input deck, geometry and all derived field bytes against provenance.
    FireField(json.loads(Path("configs/scenarios/gaehwasan.json").read_bytes()))
    report["fds_provenance_verified"] = True
    started = (out/"phases.json").stat().st_mtime - report["phases"]["total_s"]
    caches = list(Path("output/gaehwasan/decision_cache").rglob("*.json"))
    caches += list(Path("output/gaehwasan/private_profile_cache").rglob("*.json"))
    report["cache_audit"] = {"method": "새 추론 금지 실행과 파일 수정 시각 검사", "file_count": len(caches),
        "modified_since_simulation_start": [str(p) for p in caches if p.stat().st_mtime > started]}
    report["active_result_modified_since_simulation_start"] = Path("output/gaehwasan/results.json").stat().st_mtime > started
    kst = timezone(timedelta(hours=9))
    report["measurement_window_kst"] = {
        "start": datetime.fromtimestamp(started, kst).isoformat(),
        "end": datetime.fromtimestamp((out/"phases.json").stat().st_mtime, kst).isoformat()}
    report["working_tree_changes_since_measurement_start"] = [
        {"path": str(p), "modified_at_kst": datetime.fromtimestamp(p.stat().st_mtime, kst).isoformat()}
        for p in Path("src/persona_evac").rglob("*.py") if p.stat().st_mtime > started]
    report["qwen_fresh_sample"] = qwen_summary(samples["records"], samples["groups"], report["configuration"]["decision_workers"])
    report["qwen_fresh_sample"]["interruption"] = "20건 성공 후 로컬 서버 재시작으로 4건 연결 오류. 성공한 20건만 지연 통계에 사용했다."
    checks = {k:v for k,v in report["baseline_comparison"].items() if k != "source"}
    report["validation_passed"] = (all(checks.values()) and report["source_metadata_validation"]["manifest_matches"]
        and report["source_metadata_validation"]["input_hash_matches"]
        and not report["cache_audit"]["modified_since_simulation_start"]
        and not report["active_result_modified_since_simulation_start"])
    report["validation_scope"] = "측정 당시 실행의 궤적·판단·원본 출처 일치 및 운영 결과·캐시 보존 검사. 현재 디스크 코드와의 동일성은 code_unchanged로 별도 표시한다."
    report["environment_change_note"] = "측정 프로세스가 불러온 코드로 전체 계산을 완료했다. 도중 station_dynamics.py, 이후 private_decisions.py 및 Ollama 서버 설정이 바뀌어 이 수치를 변경 후 환경의 재측정으로 해석하지 않는다."
    write_json(out/"report.json", report)
    write_json(out/"progress.json", {"phase": "완료", "report": str(out/"report.json"), "validation_passed": report["validation_passed"]})
    assert report["validation_passed"], "계측 결과 또는 출처 검증이 일치하지 않습니다."
    print(json.dumps({"phase":"저장된 측정 보고서 검증 완료", "qwen":report["qwen_fresh_sample"],
                      "source_validation":report["source_metadata_validation"], "cache_audit":report["cache_audit"]},ensure_ascii=False),flush=True)


def self_test():
    # Nested wall intervals: root=10, parent=6, child=2 -> exclusive 4+4+2=10.
    values = iter([0., 1., 2., 4., 7., 10.])
    timer = PhaseTimer(lambda: next(values))
    with timer.span("root", "other"):
        with timer.span("parent", "route"):
            with timer.span("child", "sense"):
                pass
    report = timer.report()
    assert report["total_s"] == 10.
    assert {r["category"]: r["seconds"] for r in report["categories"]} == {
        "other": 4., "route": 4., "sense": 2.}
    with ThreadPoolExecutor(max_workers=1) as pool:
        def worker():
            with timer.span("worker", "qwen"):
                return 7
        assert pool.submit(worker).result() == 7
    assert timer.report() == report
    try:
        values = iter([0., 1.])
        timer = PhaseTimer(lambda: next(values))
        with timer.span("error", "other"):
            raise ValueError("검사")
    except ValueError:
        pass
    assert timer.report()["total_s"] == 1.
    summary = qwen_summary(
        [{"status":"valid", "wall_s":1.}, {"status":"valid", "wall_s":2.},
         {"status":"failed", "wall_s":.001}],
        [{"count":2, "wall_s":2.}, {"count":1, "wall_s":.001}], 2)
    assert summary["valid_count"] == 2 and summary["failed_count"] == 1
    assert summary["request_min_s"] == 1. and summary["request_median_s"] == 1.5
    assert summary["valid_batch_wall_total_s"] == 2.
    print("중첩 시간·작업 스레드 중복 제외·예외 시 계측·실패 표본 제외 검사 통과", flush=True)


def choose_samples(observations, count):
    # Cover short and long real requests, spread over the entire simulation.
    from persona_evac.llm.private_decisions import project_private
    ordered = sorted(observations, key=lambda o: len(json.dumps(project_private(o), ensure_ascii=False)))
    count = min(count, len(ordered))
    if count <= 1:
        return ordered[:count]
    return [ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/gaehwasan/performance_profile"))
    parser.add_argument("--duration", type=float, default=600.)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--qwen-samples", type=int, default=24)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--finalize-existing", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.finalize_existing:
        finalize_existing(args.finalize_existing)
        return
    if args.duration < 5 or args.qwen_samples < 0 or args.sample_every < 0:
        parser.error("계산 시간은 5초 이상, 표본 수와 표본 간격은 0 이상이어야 합니다.")

    import jupedsim as jps
    from persona_evac.integration import station_dynamics as dynamics
    from persona_evac.integration import private_agents as controller
    from persona_evac.integration import personal_space as space
    from persona_evac.integration.fds_environment import FireField
    from persona_evac.integration.station import StationRequest, simulate_station
    from persona_evac.llm import private_decisions as choices
    from persona_evac.llm.ollama import LocalOllama
    from persona_evac.station_artifacts import station_signature

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    signature = station_signature()
    protected_paths = [Path(p) for p in (
        "output/gaehwasan/results.json", "output/gaehwasan/persona_source.json",
        "configs/scenarios/gaehwasan.json", "configs/fds/gaehwasan.fds",
        "output/fds/gaehwasan/derived/fields.npz", "output/fds/gaehwasan/derived/metadata.json")]
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
    cache_paths = list(Path("output/gaehwasan/decision_cache").rglob("*.json"))
    cache_paths += list(Path("output/gaehwasan/private_profile_cache").rglob("*.json"))
    cache_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cache_paths}
    write_json(out/"initial_hashes.json", {"protected": protected, "caches": cache_hashes})
    local = LocalOllama()
    local.settings["num_ctx"] = 4096
    before_ps = model_state(local)
    original_generate = LocalOllama.generate

    def refuse_inference(*_args, **_kwargs):
        raise CacheMiss("계측 재현 중 캐시 미스가 발생했습니다. 운영 캐시에 쓰지 않고 중단합니다.")

    LocalOllama.generate = refuse_inference
    timer = PhaseTimer()
    timer.wrap(jps.Simulation, "iterate", "jupedsim")
    timer.wrap(dynamics, "initialize_station", "initialization")
    timer.wrap(controller.PrivateController, "__init__", "initialization")
    timer.wrap(FireField, "__init__", "initialization")
    timer.wrap(controller.PrivateController, "tick", "controller_other")
    timer.wrap(controller.PrivateController, "decide_batch", "qwen_batch_wait")
    timer.wrap(controller.PrivateController, "observation", "observation")
    timer.wrap(controller.PrivateController, "visible_helper", "observation")
    timer.wrap(space.OccludedSensor, "sense", "observation")
    timer.wrap(controller.PrivateController, "update_navigation", "routing")
    timer.wrap(space.PersonalSpace, "options", "routing")
    timer.wrap(space.PersonalSpace, "shortest", "routing")
    timer.wrap(controller, "resolve_private", "routing")
    for name in ("plan_yield", "plan_cluster_detour"):
        timer.wrap(dynamics, name, "routing")
    for name in ("replay", "stair_observations"):
        timer.wrap(FireField, name, "replay_output")

    sampled = cProfile.Profile()
    sampling = [False]
    sample_count = defaultdict(int)
    for owner, name in ((space.OccludedSensor, "sense"),
                        (controller.PrivateController, "update_navigation")):
        original = getattr(owner, name)
        def install(original=original, name=name):
            @functools.wraps(original)
            def call(*a, **kw):
                sample_count[name] += 1
                active = args.sample_every > 0 and sample_count[name] % args.sample_every == 0 and not sampling[0]
                if active:
                    sampling[0] = True
                    sampled.enable()
                try:
                    return original(*a, **kw)
                finally:
                    if active:
                        sampled.disable()
                        sampling[0] = False
            return call
        setattr(owner, name, install())

    observations = []
    batches = []
    original_batch = controller.PrivateController.decide_batch
    def capture_batch(self, batch):
        observations.extend(copy.deepcopy(batch))
        started = time.perf_counter()
        result = original_batch(self, batch)
        batches.append({"time_s": batch[0]["time_s"], "count": len(batch),
                        "wall_s": time.perf_counter()-started})
        return result
    controller.PrivateController.decide_batch = capture_batch

    heartbeat_stop = threading.Event()
    started = time.perf_counter()
    def heartbeat():
        while not heartbeat_stop.wait(30):
            write_json(out/"progress.json", {"phase": "캐시를 사용한 전체 대피 계산 계측",
                "wall_s": time.perf_counter()-started, "decision_count": len(observations),
                "last_decision_time_s": batches[-1]["time_s"] if batches else None})
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        source = json.loads(Path("output/gaehwasan/persona_source.json").read_bytes())
        config = json.loads(Path("configs/scenarios/gaehwasan.json").read_bytes())
        with timer.span("simulate_station", "other"):
            result = simulate_station(source, config, StationRequest(agent_count=args.agents,
                duration_s=args.duration, behavior_mode="qwen", fire_mode="fds", experiment_seed=20261027))
    finally:
        heartbeat_stop.set()
        thread.join()
        LocalOllama.generate = original_generate
        write_json(out/"phases.json", timer.report())
        sampled.dump_stats(str(out/"sampled.pstats"))

    baseline = json.loads(Path("output/gaehwasan/results.json").read_bytes())
    # run_station/API attach this provenance field after simulate_station returns.
    result["source"]["artifact_sha256"] = protected["output/gaehwasan/persona_source.json"]
    compare_keys = ("summary", "agents", "passages", "scenario", "source", "navigation_updates", "noncompletion", "movement_diagnostics")
    comparison = {k: digest(result.get(k)) == digest(baseline.get(k)) for k in compare_keys}
    decision_keys = ("agent_id", "time_s", "raw_model_choice", "model_choice", "final_target", "strategy_costs",
                     "events", "observation", "executed_target", "executed_goal", "execution_source")
    def decision_digest(r):
        return digest([{k: d.get(k) for k in decision_keys} for d in r["decisions"]])
    comparison["decisions"] = decision_digest(result) == decision_digest(baseline)
    stats = pstats.Stats(sampled)
    top = [{"file": Path(f).name, "line": ln, "function": fn, "primitive_calls": cc,
            "calls": nc, "self_s": tt, "cumulative_s": ct}
           for (f, ln, fn), (cc, nc, tt, ct, _) in stats.stats.items()]
    write_json(out/"sampled_functions.json", sorted(top, key=lambda r: -r["self_s"])[:80])
    report = {"environment": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
               "jupedsim": jps.__version__, "model": local.model, "model_digest": local.digest,
               "ollama_ps_before": before_ps},
        "configuration": {"agents": args.agents, "duration_s": args.duration,
            "seed": 20261027, "decision_workers": int(os.environ.get("PERSONA_EVAC_DECISION_WORKERS", "2")),
            "sample_every": args.sample_every, "qwen_sample_count_requested": args.qwen_samples},
        "phases": timer.report(), "performance": result["performance"], "summary": result["summary"],
        "decision_count": len(result["decisions"]), "decision_batches": batches,
        "baseline_comparison": comparison, "implementation_sha256": signature,
        "notes": ["전체 실행은 운영 판단 캐시를 사용한다. 새 추론을 시도하면 중단한다.",
                  "구간 비율은 중첩을 제외한 주 스레드 경과 시간이다.",
                  "관측 시간에는 센서의 FDS 결과 표본화와 가시성 검사가 포함된다.",
                  "JuPedSim은 iterate 안의 이동·근접 탐색·내부 경로 계산을 포함한다.",
                  "세부 함수 표본은 전체 관측·경로 호출 200회 중 1회이며 전수 시간이 아니다."]}
    write_json(out/"report.json", report)
    print(json.dumps({"phase": "전체 대피 계산 계측 완료", "phases": report["phases"]["categories"],
                      "comparison": comparison}, ensure_ascii=False), flush=True)

    selected = choose_samples(observations, args.qwen_samples)
    write_json(out/"qwen_sample_inputs.json", selected)
    records, groups = [], []
    def fresh(o):
        started = time.perf_counter()
        try:
            payload, metric = local.generate(choices.PROMPT, [choices.project_private(o)],
                                             choices.private_schema([o]), 768, prompt_batch_size=128)
            choices.validate_private(payload, [o])
            return {"agent_id": o["agent_id"], "time_s": o["time_s"], "events": o["events"],
                    "input_chars": len(json.dumps(choices.project_private(o), ensure_ascii=False)),
                    "status": "valid", **metric}
        except Exception as exc:
            return {"agent_id": o["agent_id"], "time_s": o["time_s"], "status": "failed",
                    "wall_s": time.perf_counter()-started, "error": str(exc)}
    workers = report["configuration"]["decision_workers"]
    for start in range(0, len(selected), workers):
        group = selected[start:start+workers]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(fresh, group))
        wall = time.perf_counter()-started
        records.extend(rows)
        groups.append({"count": len(group), "wall_s": wall})
        write_json(out/"qwen_samples.json", {"records": records, "groups": groups})
        write_json(out/"progress.json", {"phase": "Qwen 캐시를 우회한 실제 추론 표본 측정",
                                        "completed": len(records), "total": len(selected)})
        print(json.dumps({"phase": "Qwen 실제 추론 표본", "completed": len(records),
                          "total": len(selected), "batch_wall_s": wall}, ensure_ascii=False), flush=True)
    report["qwen_fresh_sample"] = qwen_summary(records, groups, workers)
    report["qwen_fresh_sample"]["ollama_ps_after"] = model_state(local)
    report["protected_files_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items())
    report["cache_files_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in cache_hashes.items())
    report["cache_file_count_unchanged"] = len(cache_paths) == len(list(Path("output/gaehwasan/decision_cache").rglob("*.json"))) + len(list(Path("output/gaehwasan/private_profile_cache").rglob("*.json")))
    report["code_unchanged"] = station_signature() == signature
    write_json(out/"report.json", report)
    write_json(out/"progress.json", {"phase": "완료", "report": str(out/"report.json")})
    assert all((report[k] for k in ("protected_files_unchanged", "cache_files_unchanged", "cache_file_count_unchanged", "code_unchanged")))
    if args.agents == 100 and args.duration == 600:
        assert all(comparison.values()), comparison
    print(json.dumps({"phase": "전체 측정 완료", "report": str(out/"report.json"),
                      "qwen": report["qwen_fresh_sample"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
