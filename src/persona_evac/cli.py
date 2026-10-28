from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_evac.integration.pyfds_adapter import inspect_pyfds_checkout, validate_fds_deck
from persona_evac.integration.pyfds_results import materialize_pyfds_results
from persona_evac.integration.runner import run_experiment
from persona_evac.persona import fetch_nemotron_snapshot
from persona_evac.empirical.julich_parser import prepare_julich_features
from persona_evac.integration.station import run_station


def main() -> None:
    parser = argparse.ArgumentParser(prog="persona-evac")
    commands = parser.add_subparsers(dest="command", required=True)
    station = commands.add_parser("run-gaehwasan", help="개화산역 도면 기반 대피 계산")
    station.add_argument("--source", type=Path, default=Path("output/gaehwasan/persona_source.json"))
    station.add_argument("--output", type=Path, default=Path("output/gaehwasan/results.json"))
    station.add_argument("--agent-count", type=int, default=100)
    station.add_argument("--fire", action="store_true")
    station.add_argument("--qwen", action="store_true")
    station.add_argument('--experiment-seed',type=int,default=20261027)
    station.add_argument('--duration-s',type=float,default=600)
    station.add_argument('--profile-method',choices=['keyword_candidates','full_persona'])
    export_station = commands.add_parser("export-gaehwasan-fire", help="개화산역 FDS 결과 변환")
    run = commands.add_parser("run", help="기존 T자 복도 데이터 결합 실험 실행")
    run.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("inspect-pyfds", help="고정된 pyFDS-Evac 연결 상태 확인")
    inspect.add_argument("--deck", type=Path)
    population = commands.add_parser("prepare-station-personas", help="역사 대피용 Nemotron 페르소나 성향 준비")
    population.add_argument("--snapshot", type=Path, default=Path("data/raw/nemotron_personas_100.jsonl"))
    population.add_argument("--empirical", type=Path, default=Path("data/processed/julich_features.csv"))
    population.add_argument("--output", type=Path, default=Path("output/gaehwasan/persona_source.json"))
    fetch = commands.add_parser("fetch-nemotron", help="고정 시드의 중복 없는 Nemotron 자료 추출")
    fetch.add_argument("--output", type=Path, default=Path("data/raw/nemotron_personas_10.jsonl"))
    fetch.add_argument("--count", type=int, default=10)
    fetch.add_argument("--seed", type=int, default=20260916)
    julich = commands.add_parser("prepare-julich", help="Jülich S1 압축 자료에서 경험적 특성 추출")
    julich.add_argument("--zip", type=Path, required=True)
    julich.add_argument("--output", type=Path, default=Path("data/processed/julich_features.csv"))
    import_pyfds = commands.add_parser("import-pyfds", help="실제 pyFDS 결과와 페르소나 자료 결합")
    import_pyfds.add_argument("--run-dir", type=Path, required=True)
    import_pyfds.add_argument("--latest-dir", type=Path)
    args = parser.parse_args()
    if args.command == "run-gaehwasan":
        print(run_station(args.source, Path("configs/scenarios/gaehwasan.json"), args.output,
                          qwen=args.qwen, agent_count=args.agent_count, fire=args.fire,
                          experiment_seed=args.experiment_seed,duration_s=args.duration_s,profile_method=args.profile_method))
    elif args.command == "export-gaehwasan-fire":
        from persona_evac.integration.fds_environment import export_fds
        directory = Path("output/fds/gaehwasan")
        export_fds(directory, directory / "derived", Path("configs/scenarios/gaehwasan.json"),
                   Path("configs/fds/gaehwasan.fds"))
    elif args.command == "run":
        print(run_experiment(args.config))
    elif args.command == "inspect-pyfds":
        status = inspect_pyfds_checkout()
        if args.deck:
            validate_fds_deck(args.deck)
            status["deck_valid"] = True
        print(json.dumps(status, ensure_ascii=False, indent=2))
    elif args.command == "prepare-station-personas":
        from persona_evac.persona.station_population import prepare_population
        print(prepare_population(args.snapshot, args.empirical, args.output))
    elif args.command == "fetch-nemotron":
        print(fetch_nemotron_snapshot(args.output, args.count, args.seed))
    elif args.command == "prepare-julich":
        print(prepare_julich_features(args.zip, args.output))
    elif args.command == "import-pyfds":
        print(materialize_pyfds_results(args.run_dir, args.latest_dir))


if __name__ == "__main__":
    main()
