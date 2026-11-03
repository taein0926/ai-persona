"""고정한 입력과 판단 캐시로 관측·경로 최적화 전후를 비교한다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def manifest(paths, root):
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in sorted(paths) if p.is_file()}


def freeze(root, suite):
    suite.mkdir(parents=True, exist_ok=False)
    for name in ("src", "configs"):
        shutil.copytree(root/name, suite/"before"/name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(suite/"before", suite/"after")
    for name in ("compare_observation_optimization.py", "profile_evacuation.py"):
        target = suite/"tools"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/"scripts"/name, target)
    inputs = suite/"inputs"
    for name in ("persona_source.json", "results.json"):
        target = inputs/"gaehwasan"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/"output/gaehwasan"/name, target)
    for name in ("private_profile_cache", "decision_cache/private_agents_final_v2"):
        shutil.copytree(root/"output/gaehwasan"/name, inputs/"gaehwasan"/name)
    shutil.copytree(root/"output/fds/gaehwasan/derived", inputs/"fds/gaehwasan/derived")
    frozen = manifest(inputs.rglob("*"), inputs)
    save(suite/"input_hashes.json", frozen)
    save(suite/"before/code_hashes.json", manifest((suite/"before").rglob("*"), suite/"before"))
    save(suite/"configuration.json", {"agent_count":100, "duration_s":600, "experiment_seed":20261027,
        "fire":"기존 FDS 결과", "behavior":"Qwen 판단 캐시", "decision_workers":2,
        "prompt_batch_size":128, "new_inference_allowed":False, "inputs_sha256":sha(suite/"input_hashes.json"),
        "created_at_utc":datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"phase":"비교용 코드·입력 고정 완료", "suite":str(suite), "input_files":len(frozen)},ensure_ascii=False),flush=True)


def semantic_hashes(result):
    """시간 계측·출처 서명 외의 전체 결과를 비교한다."""
    hashes = {}
    for key, value in result.items():
        if key in {"performance", "implementation_sha256", "computation"}:
            continue
        if key == "behavior":
            value = {k:v for k,v in value.items() if k not in {"timings", "inference_attempts"}}
        if key == "decisions":
            # Execution metadata was added after the preserved original artifact.
            value = [{k:v for k,v in d.items() if k not in {"wall_s", "prompt_batch_size"}
                      and not (k=="runtime_profile" and v is None)} for d in value]
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        hashes[key] = hashlib.sha256(encoded).hexdigest()
    return hashes


def run(suite, variant, duration):
    import jupedsim as jps
    from profile_evacuation import PhaseTimer
    from persona_evac.integration import station_dynamics as dynamics
    from persona_evac.integration import private_agents as controller
    from persona_evac.integration import personal_space as space
    from persona_evac.integration.fds_environment import FireField
    from persona_evac.integration.station import StationRequest, simulate_station
    from persona_evac.llm.ollama import LocalOllama
    from persona_evac.station_artifacts import station_signature

    out = suite/variant/"measurement"
    out.mkdir(exist_ok=False)
    work = suite/variant
    inputs = suite/"inputs"
    input_hashes = json.loads((suite/"input_hashes.json").read_bytes())
    assert manifest(inputs.rglob("*"), inputs) == input_hashes, "고정 입력 또는 캐시가 변경되었습니다."
    code_paths = [p for name in ("src", "configs") for p in (work/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    code = manifest(code_paths, work)
    save(out/"code_hashes.json", code)
    signature = station_signature()
    class UnexpectedInference(BaseException):
        pass
    def no_inference(*_args, **_kwargs):
        raise UnexpectedInference("기록된 판단 캐시가 없습니다. 새 추론 없이 비교를 중단합니다.")
    LocalOllama.generate = no_inference
    timer = PhaseTimer()
    for owner, name, category in (
        (jps.Simulation,"iterate","jupedsim"), (dynamics,"initialize_station","initialization"),
        (controller.PrivateController,"__init__","initialization"), (FireField,"__init__","initialization"),
        (controller.PrivateController,"tick","controller_other"),
        (controller.PrivateController,"decide_batch","qwen_batch_wait"),
        (controller.PrivateController,"observation","observation"),
        (controller.PrivateController,"visible_helper","observation"),
        (space.OccludedSensor,"sense","observation"),
        (controller.PrivateController,"update_navigation","routing"),
        (space.PersonalSpace,"options","routing"), (space.PersonalSpace,"shortest","routing"),
        (controller,"resolve_private","routing"), (FireField,"replay","replay_output"),
        (FireField,"stair_observations","replay_output"),
        (dynamics,"plan_yield","routing"),
        (dynamics,"plan_cluster_detour","routing")):
        timer.wrap(owner,name,category)
    progress = {"simulation_time_s":0., "decisions":0}
    original_tick = controller.PrivateController.tick
    def tick(self, now):
        progress.update(simulation_time_s=now, decisions=len(self.decisions))
        return original_tick(self,now)
    controller.PrivateController.tick = tick
    stop = threading.Event()
    start = time.perf_counter()
    def heartbeat():
        while not stop.wait(30):
            save(out/"progress.json", {"phase":"대피 계산 계측 중", "wall_s":time.perf_counter()-start, **progress})
    thread = threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    source = json.loads((inputs/"gaehwasan/persona_source.json").read_bytes())
    config = json.loads((work/"configs/scenarios/gaehwasan.json").read_bytes())
    cpu_start = time.process_time()
    try:
        with timer.span("simulate_station","other"):
            result = simulate_station(source, config, StationRequest(agent_count=100,duration_s=duration,
                behavior_mode="qwen",fire_mode="fds",experiment_seed=20261027))
    finally:
        stop.set()
        thread.join()
        save(out/"phases.json",timer.report())
    cpu_s = time.process_time()-cpu_start
    result["source"]["artifact_sha256"] = sha(inputs/"gaehwasan/persona_source.json")
    result["input_config_sha256"] = sha(work/"configs/scenarios/gaehwasan.json")
    result["implementation_sha256"] = signature
    result["computation"] = {"code_sha256":signature,"kind":"measured_recomputation_with_preserved_decisions"}
    baseline = json.loads((inputs/"gaehwasan/results.json").read_bytes())
    reference_hashes = semantic_hashes(baseline)
    hashes = semantic_hashes(result)
    # The preserved artifact may lack diagnostic-only behavior fields added later.
    primary = ("agents","passages","summary","scenario","navigation_updates","movement_diagnostics",
               "noncompletion","decisions","fire","agent_policy","termination","replay_duration_s","source")
    comparison = {k:hashes.get(k)==reference_hashes.get(k) for k in primary}
    assert manifest(code_paths,work) == code, "고정한 계산 코드가 실행 중 변경되었습니다."
    assert manifest(inputs.rglob("*"),inputs) == input_hashes, "고정 입력 또는 캐시가 실행 중 변경되었습니다."
    assert result["performance"]["llm_call_count"] == 0
    assert result["behavior"]["fallback_count"] == 0
    report = {"variant":variant, "configuration":json.loads((suite/"configuration.json").read_bytes()),
        "duration_s":duration, "phases":timer.report(), "cpu_time_s":cpu_s,
        "performance":result["performance"], "summary":result["summary"],
        "semantic_hashes":hashes, "reference_comparison":comparison,
        "code_unchanged_during_run":True,"inputs_and_caches_unchanged":True,
        "implementation_sha256":signature,"jupedsim_version":jps.__version__}
    save(out/"report.json",report)
    with gzip.open(out/"result.json.gz","wt",encoding="utf-8",compresslevel=1) as stream:
        json.dump(result,stream,ensure_ascii=False,separators=(",",":"))
    save(out/"progress.json", {"phase":"완료", "wall_s":timer.report()["total_s"],"comparison":comparison})
    print(json.dumps({"phase":"계측 완료","variant":variant,"phases":timer.report()["categories"],
        "reference_comparison":comparison},ensure_ascii=False),flush=True)
    if duration == 600:
        assert all(comparison.values()), "원본 궤적 또는 판단 결과가 다릅니다."


def compare(suite):
    before = json.loads((suite/"before/measurement/report.json").read_bytes())
    after = json.loads((suite/"after/measurement/report.json").read_bytes())
    assert before["configuration"] == after["configuration"]
    assert before["duration_s"] == after["duration_s"] == 600
    assert before["semantic_hashes"] == after["semantic_hashes"], "시간·코드 서명 외에 결과가 다릅니다."
    assert all(before["reference_comparison"].values()) and all(after["reference_comparison"].values())
    btime, atime = before["phases"]["total_s"], after["phases"]["total_s"]
    b = {r["category"]:r["seconds"] for r in before["phases"]["categories"]}
    a = {r["category"]:r["seconds"] for r in after["phases"]["categories"]}
    report = {"configuration":before["configuration"], "before_s":btime,"after_s":atime,
        "speedup":btime/atime,"reduction_percent":100*(1-atime/btime),
        "all_semantic_result_fields_identical":True, "all_reference_checks_passed":True,
        "before_cpu_s":before["cpu_time_s"],"after_cpu_s":after["cpu_time_s"],
        "categories":[{"category":k,"before_s":b.get(k,0),"after_s":a.get(k,0),
                       "reduction_percent":100*(1-a.get(k,0)/b[k]) if b.get(k) else None} for k in b]}
    save(suite/"comparison.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def recheck(suite,variant):
    out=suite/variant/"measurement"
    report=json.loads((out/"report.json").read_bytes())
    with gzip.open(out/"result.json.gz","rt",encoding="utf-8") as stream:
        result=json.load(stream)
    reference=json.loads((suite/"inputs/gaehwasan/results.json").read_bytes())
    measured=semantic_hashes(result);original=semantic_hashes(reference)
    report["semantic_hashes"]=measured
    report["reference_comparison"]={k:measured.get(k)==original.get(k) for k in report["reference_comparison"]}
    report["comparison_metadata_note"]="판단 비교에서는 wall_s, 실행 배치 크기 메타데이터, 나중에 추가된 null runtime_profile 필드만 제외한다. 입력·모델 선택·기억·관측·실행 목표는 모두 비교한다."
    save(out/"report.json",report)
    assert all(report["reference_comparison"].values()),report["reference_comparison"]
    save(out/"progress.json",{"phase":"완료","wall_s":report["phases"]["total_s"],"comparison":report["reference_comparison"]})
    print(json.dumps({"phase":"저장된 결과 재검증 완료","variant":variant,
        "reference_comparison":report["reference_comparison"]},ensure_ascii=False),flush=True)


def verify_reference(suite):
    """추가 변경을 보존한 현재 결과와 비교한다. 새 전후 시간 비교로 표시하지 않는다."""
    out=suite/"after/measurement"
    measured=json.loads((out/"report.json").read_bytes())
    reference=json.loads((suite/"inputs/gaehwasan/results.json").read_bytes())
    original=semantic_hashes(reference)
    identical=measured["semantic_hashes"]==original
    assert identical and all(measured["reference_comparison"].values())
    report={"configuration":measured["configuration"],"all_semantic_result_fields_identical":identical,
        "all_reference_checks_passed":True,"measured_s":measured["phases"]["total_s"],
        "measurement_method":"고정한 현재 결과와 추가 변경 후 100명·600초 계산의 전체 결과 비교. 새 추론 없음. 새 전후 성능 비교는 수행하지 않음."}
    save(suite/"comparison.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def activate(suite):
    from persona_evac.agent_policy import ROOT
    from persona_evac.station_artifacts import station_signature, policy_matches
    comparison=json.loads((suite/"comparison.json").read_bytes())
    assert comparison["all_semantic_result_fields_identical"] and comparison["all_reference_checks_passed"]
    out=suite/"after/measurement"
    code=json.loads((out/"code_hashes.json").read_bytes())
    assert all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in code.items()), "현재 코드가 검증한 최적화 사본과 다릅니다."
    target=ROOT/"output/gaehwasan/results.json"
    original=suite/"inputs/gaehwasan/results.json"
    previous_hash=sha(target)
    assert previous_hash==sha(original), "기준 입력을 고정한 뒤 현재 결과가 변경되어 덮어쓰지 않습니다."
    with gzip.open(out/"result.json.gz","rt",encoding="utf-8") as stream:
        result=json.load(stream)
    assert result["implementation_sha256"]==station_signature()
    assert result["source"]["artifact_sha256"]==sha(ROOT/"output/gaehwasan/persona_source.json")
    assert result["input_config_sha256"]==sha(ROOT/"configs/scenarios/gaehwasan.json")
    assert policy_matches(result)
    temp=target.with_suffix('.observation_optimization.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,separators=(',',':'))
    temp.replace(target)
    report={"status":"검증한 최적화 결과 적용 완료", "previous_result_sha256":previous_hash,
        "preserved_original":str(original),"active_result":str(target),"active_result_sha256":sha(target),
        "implementation_sha256":result["implementation_sha256"],"new_inference_calls":0,
        "judgments_and_trajectories_preserved":True}
    save(suite/"activation.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","run","compare","recheck","activate","verify-reference"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--variant",choices=("before","after"))
    parser.add_argument("--duration",type=float,default=600)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(Path.cwd(),args.suite)
    elif args.action in {"run","recheck"}:
        if not args.variant:parser.error("실행에는 --variant가 필요합니다.")
        if args.action=="run":run(args.suite,args.variant,args.duration)
        else:recheck(args.suite,args.variant)
    elif args.action=="activate":
        activate(args.suite)
    elif args.action=="verify-reference":
        verify_reference(args.suite)
    else:
        compare(args.suite)


if __name__ == "__main__":
    main()
