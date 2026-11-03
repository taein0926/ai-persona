"""이미지·문서 없이 고정한 시뮬레이션 데이터만 묶고 검증·복원한다."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_FILES = (
    "data/raw/nemotron_personas_10.jsonl",
    "data/raw/nemotron_personas_10.metadata.json",
    "data/raw/nemotron_personas_100.jsonl",
    "data/raw/nemotron_personas_100.metadata.json",
    "data/raw/julich-social-groups-s1.zip",
    "data/processed/julich_features.csv",
    "data/processed/julich_features.metadata.json",
    "output/gaehwasan/persona_source.json",
    "output/gaehwasan/results.json",
    "output/gaehwasan/private_verification/activation.json",
    "output/fds/gaehwasan/derived/fields.npz",
    "output/fds/gaehwasan/derived/metadata.json",
    "output/latest/results.json",
    "output/latest/assignment_manifest.parquet",
    "output/t-junction-mvp-openai/seed-1/pyfds_results.json",
)
CACHE_DIRS = (
    "output/gaehwasan/private_profile_cache",
    "output/gaehwasan/decision_cache",
)
CONFIG_FILES = (
    "configs/scenarios/gaehwasan.json",
    "configs/fds/gaehwasan.fds",
    "configs/agents/gaehwasan_prior.json",
    "configs/agents/private_agent_policy.json",
)
ARCHIVE = "simulation-data.zip"
PART_BYTES = 40 * 1024 * 1024
SOURCE_FALLBACKS = {
    "output/latest/results.json": "output/t-junction-mvp-openai/seed-1/pyfds_results.json",
    "output/latest/assignment_manifest.parquet": "output/t-junction-mvp-openai/seed-1/assignment_manifest.parquet",
}


def packing_source(root, name):
    path = destination(root, name)
    if not path.exists() and name in SOURCE_FALLBACKS:
        return destination(root, SOURCE_FALLBACKS[name])
    return path


def publish(source, target):
    # Windows 전용 임시 폴더의 권한을 최종 파일로 옮기지 않는다.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as content:
                shutil.copyfileobj(content, stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(target)


@contextmanager
def open_archive(bundle, manifest):
    metadata = manifest["archive"]
    if manifest["version"] == 1:
        archive = bundle / ARCHIVE
        if archive.stat().st_size != metadata["size"] or sha(archive) != metadata["sha256"]:
            raise ValueError("실행 데이터 압축 파일의 크기 또는 해시가 다릅니다.")
        yield archive
        return
    parts = metadata.get("parts", [])
    if not parts or [item["path"] for item in parts] != [
        f"{ARCHIVE}.part{index:03d}" for index in range(1, len(parts) + 1)
    ]:
        raise ValueError("실행 데이터 압축 파일의 분할 목록이 올바르지 않습니다.")
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        for item in parts:
            path = bundle / item["path"]
            if path.stat().st_size != item["size"]:
                raise ValueError(f"실행 데이터 압축 파일 조각의 크기가 다릅니다: {item['path']}")
            part_digest = hashlib.sha256()
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    part_digest.update(block)
                    digest.update(block)
                    archive.write(block)
            if part_digest.hexdigest() != item["sha256"]:
                raise ValueError(f"실행 데이터 압축 파일 조각의 해시가 다릅니다: {item['path']}")
        if archive.tell() != metadata["size"] or digest.hexdigest() != metadata["sha256"]:
            raise ValueError("실행 데이터 압축 파일 전체의 크기 또는 해시가 다릅니다.")
        archive.seek(0)
        yield archive


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def allowed(name):
    return name in FIXED_FILES or (
        any(name.startswith(directory + "/") for directory in CACHE_DIRS)
        and name.endswith(".json")
    )


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"허용되지 않은 데이터 경로: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"프로젝트 밖을 가리키는 데이터 경로: {name}")
    if os.name == "nt":
        # 긴 캐시 파일명도 Windows의 기본 260자 제한에 걸리지 않게 한다.
        absolute = str(path.absolute())
        if not absolute.startswith("\\\\?\\"):
            absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                        else "\\\\?\\" + absolute)
        return Path(absolute)
    return path


def pack(root):
    names = list(FIXED_FILES)
    for directory in CACHE_DIRS:
        names.extend(p.relative_to(root).as_posix() for p in (root / directory).rglob("*.json"))
    names = sorted(set(names))
    for name in names:
        if not allowed(name) or not packing_source(root, name).is_file():
            raise ValueError(f"필수 데이터가 없거나 포함할 수 없습니다: {name}")
    bundle = root / "runtime"
    bundle.mkdir(exist_ok=True)
    files = []
    with tempfile.TemporaryDirectory(dir=bundle) as temporary:
        archive = Path(temporary) / ARCHIVE
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for name in names:
                data = packing_source(root, name).read_bytes()
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                zipped.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                files.append({"path": name, "size": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
        population = json.loads((root / "output/gaehwasan/persona_source.json").read_text(encoding="utf-8"))
        parts = []
        with archive.open("rb") as stream:
            while block := stream.read(PART_BYTES):
                name = f"{ARCHIVE}.part{len(parts) + 1:03d}"
                part = Path(temporary) / name
                part.write_bytes(block)
                parts.append({"path": name, "size": len(block), "sha256": sha(part)})
        manifest = {
            "version": 2,
            "archive": {"path": ARCHIVE, "size": archive.stat().st_size, "sha256": sha(archive)},
            "model": population.get("provenance", {}).get("model"),
            "prepared_model_digest": population.get("provenance", {}).get("mapping_settings", {}).get("model_digest"),
            "purpose": "현재 두 화면의 결과 재생, 개화산역 기존 FDS를 이용한 대피 재계산",
            "excluded": "이미지·PDF·PPT·비밀 설정·모델 가중치·FDS 원시 출력·중간 검증 결과·서버 로그",
            "configs": {name: sha(root / name) for name in CONFIG_FILES},
            "files": files,
        }
        manifest["archive"]["parts"] = parts
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        for source in [*(Path(temporary) / item["path"] for item in parts), manifest_path]:
            publish(source, bundle / source.name)
        # 새 묶음 게시 후, 이 도구가 생성한 이전 조각과 단일 파일만 정리한다.
        current = {item["path"] for item in parts}
        for old in [bundle / ARCHIVE, *bundle.glob(f"{ARCHIVE}.part[0-9][0-9][0-9]")]:
            if old.name not in current:
                old.unlink(missing_ok=True)
    return {"status": "데이터 묶음 생성 완료", "files": len(files),
            "archive_bytes": manifest["archive"]["size"], "parts": len(parts),
            "unpacked_bytes": sum(item["size"] for item in files)}


def restore(root, install=False):
    bundle = root / "runtime"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") not in (1, 2) or manifest.get("archive", {}).get("path") != ARCHIVE:
        raise ValueError("지원하지 않는 실행 데이터 형식입니다.")
    for name in CONFIG_FILES:
        if sha(root / name) != manifest["configs"].get(name):
            raise ValueError(f"고정된 입력 설정이 다릅니다: {name}")
    entries = manifest["files"]
    expected = {item["path"]: item for item in entries}
    if len(expected) != len(entries) or not set(FIXED_FILES).issubset(expected):
        raise ValueError("실행 데이터 목록이 중복되었거나 필수 파일이 없습니다.")
    missing = []
    with open_archive(bundle, manifest) as archive, zipfile.ZipFile(archive) as zipped:
        if len(zipped.namelist()) != len(expected) or set(zipped.namelist()) != set(expected):
            raise ValueError("압축 파일과 실행 데이터 목록이 다릅니다.")
        # 모든 경로·크기·해시·기존 파일을 먼저 검사하고 나서 복원한다.
        for name, item in expected.items():
            target = destination(root, name)
            if not allowed(name):
                raise ValueError(f"실행 데이터로 허용하지 않은 파일: {name}")
            if zipped.getinfo(name).file_size != item["size"]:
                raise ValueError(f"압축 항목 크기가 다릅니다: {name}")
            with zipped.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"압축 항목 해시가 다릅니다: {name}")
            if target.exists():
                if not target.is_file() or sha(target) != item["sha256"]:
                    raise ValueError(f"기존 데이터가 다릅니다. 별도 새 폴더에서 복원하세요: {name}")
            else:
                missing.append(name)
        if install:
            for name in missing:
                target = destination(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        with zipped.open(name) as source:
                            shutil.copyfileobj(source, stream)
                    except BaseException:
                        stream.close()
                        temporary.unlink(missing_ok=True)
                        raise
                temporary.replace(target)
    if missing and not install:
        raise ValueError(f"아직 복원되지 않은 데이터 {len(missing)}개가 있습니다. install을 실행하세요.")
    return {"status": "복원 완료" if install else "무결성 검증 통과",
            "files": len(expected), "restored": len(missing),
            "existing_files_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "install", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = pack(args.root.resolve()) if args.action == "pack" else restore(
            args.root.resolve(), install=args.action == "install")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"실행 데이터 처리 실패: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
