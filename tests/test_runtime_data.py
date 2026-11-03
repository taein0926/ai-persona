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
    original = (source / "runtime/manifest.json").read_bytes()
    runtime.pack(source)
    assert (source / "runtime/manifest.json").read_bytes() == original
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
    manifest = json.loads((target / "runtime/manifest.json").read_text(encoding="utf-8"))
    archive = target / "runtime" / manifest["archive"]["parts"][0]["path"]
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
    with runtime.open_archive(bundle, manifest) as archive, zipfile.ZipFile(archive) as zipped:
        contents = {name: zipped.read(name) for name in zipped.namelist()}
    last = next(reversed(contents))
    contents[last] = b"!" * len(contents[last])
    archive = bundle / manifest["archive"]["parts"][0]["path"]
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in contents.items():
            zipped.writestr(name, data)
    manifest["archive"].update(size=archive.stat().st_size, sha256=runtime.sha(archive))
    manifest["archive"]["parts"] = [{"path": archive.name, "size": archive.stat().st_size,
                                     "sha256": runtime.sha(archive)}]
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="항목 해시"):
        runtime.restore(target, install=True)
    assert not (target / "output").exists()


def test_many_parts_restore_and_missing_part_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "PART_BYTES", 256)
    _, target = prepared_bundle(tmp_path)
    bundle = target / "runtime"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    parts = manifest["archive"]["parts"]
    assert len(parts) > 1 and all(item["size"] <= 256 for item in parts)
    last = bundle / parts[-1]["path"]
    saved = last.read_bytes()
    last.unlink()
    with pytest.raises(FileNotFoundError):
        runtime.restore(target, install=True)
    assert not (target / "output").exists()
    last.write_bytes(saved)
    assert runtime.restore(target, install=True)["restored"] == len(runtime.FIXED_FILES) + 1


def test_legacy_single_archive_still_restores(tmp_path):
    _, target = prepared_bundle(tmp_path)
    bundle = target / "runtime"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    with runtime.open_archive(bundle, manifest) as joined:
        with zipfile.ZipFile(joined) as zipped:
            contents = {name: zipped.read(name) for name in zipped.namelist()}
    archive = bundle / runtime.ARCHIVE
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in contents.items():
            zipped.writestr(name, data)
    manifest["version"] = 1
    manifest["archive"] = {"path": runtime.ARCHIVE, "size": archive.stat().st_size,
                           "sha256": runtime.sha(archive)}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runtime.restore(target, install=True)["restored"] == len(runtime.FIXED_FILES) + 1


def test_missing_latest_view_uses_recorded_corridor_without_rewriting_source(tmp_path):
    source, target = prepared_bundle(tmp_path)
    for name, fallback in runtime.SOURCE_FALLBACKS.items():
        (source / name).unlink()
        original = source / fallback
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"recorded corridor data")
    runtime.pack(source)
    shutil.copytree(source / "runtime", target / "runtime", dirs_exist_ok=True)
    runtime.restore(target, install=True)
    for name in runtime.SOURCE_FALLBACKS:
        assert not (source / name).exists()
        assert (target / name).read_bytes() == b"recorded corridor data"
