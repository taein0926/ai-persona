"""배포 데이터 복원은 손상된 자료나 기존 실험을 덮어쓰지 않아야 한다."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "runtime_data", Path(__file__).resolve().parents[1] / "scripts/runtime_data.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def prepared_bundle(tmp_path):
    source = tmp_path / "source"
    for name in (*runtime.FIXED_FILES, *runtime.CONFIG_FILES):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n" if name.endswith(".json") else b"fixture\x00data")
    private = source / runtime.CACHE_DIRS[0] / "version/uuid/seed/cache.json"
    private.parent.mkdir(parents=True)
    private.write_text("{}", encoding="utf-8")
    # 참고 자료와 비밀 설정은 데이터 목록에 포함하지 않는다.
    (source / ".env").write_text("PRIVATE_VALUE=fixture", encoding="utf-8")
    (private.parent / "reference.png").write_bytes(b"local image")
    runtime.pack(source)
    target = tmp_path / "target"
    shutil.copytree(source / "runtime", target / "runtime")
    shutil.copytree(source / "configs", target / "configs")
    return source, target


def test_bundle_round_trip_is_repeatable_and_preserves_original_bytes(tmp_path):
    source, target = prepared_bundle(tmp_path)
    original = runtime.sha(source / "runtime" / runtime.ARCHIVE)
    runtime.pack(source)
    assert runtime.sha(source / "runtime" / runtime.ARCHIVE) == original
    restored = runtime.restore(target, install=True)
    assert restored["restored"] == len(runtime.FIXED_FILES) + 1
    for name in runtime.FIXED_FILES:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert runtime.restore(target, install=True)["restored"] == 0
    assert runtime.restore(target)["existing_files_preserved"]
    assert not (target / ".env").exists()
    assert not list(target.rglob("*.png"))


def test_conflicting_existing_experiment_prevents_any_install(tmp_path):
    _, target = prepared_bundle(tmp_path)
    protected = target / "output/gaehwasan/results.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"existing experiment")
    with pytest.raises(ValueError, match="기존 데이터"):
        runtime.restore(target, install=True)
    assert protected.read_bytes() == b"existing experiment"
    assert not (target / runtime.FIXED_FILES[0]).exists()


def test_corrupt_archive_and_changed_config_are_rejected(tmp_path):
    _, target = prepared_bundle(tmp_path)
    archive = target / "runtime" / runtime.ARCHIVE
    original = archive.read_bytes()
    archive.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="압축 파일"):
        runtime.restore(target, install=True)
    archive.write_bytes(original)
    (target / runtime.CONFIG_FILES[0]).write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="입력 설정"):
        runtime.restore(target, install=True)
    assert not (target / "output").exists()


@pytest.mark.parametrize("name", ("../escape.json", "/escape.json", "C:/escape.json", "output/../../escape.json"))
def test_paths_cannot_escape_project_root(tmp_path, name):
    with pytest.raises(ValueError, match="데이터 경로"):
        runtime.destination(tmp_path, name)


def test_damaged_member_is_found_before_any_file_is_written(tmp_path):
    _, target = prepared_bundle(tmp_path)
    bundle = target / "runtime"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    archive = bundle / runtime.ARCHIVE
    with zipfile.ZipFile(archive) as zipped:
        contents = {name: zipped.read(name) for name in zipped.namelist()}
    last = next(reversed(contents))
    contents[last] = b"!" * len(contents[last])
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in contents.items():
            zipped.writestr(name, data)
    manifest["archive"].update(size=archive.stat().st_size, sha256=runtime.sha(archive))
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="항목 해시"):
        runtime.restore(target, install=True)
    assert not (target / "output").exists()
