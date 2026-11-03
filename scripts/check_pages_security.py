"""추적된 소스와 실제 공개 폴더를 검사하며 비밀 값은 출력하지 않는다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

if __package__:
    from .build_pages import PUBLISHED_FILES, ROOT
else:
    from build_pages import PUBLISHED_FILES, ROOT


def validate_public_files(site):
    files=set()
    for path in site.rglob('*'):
        if path.is_symlink():
            raise ValueError('공개 폴더에 심볼릭 링크를 포함할 수 없습니다.')
        if path.is_file():files.add(path.relative_to(site).as_posix())
    if files!=PUBLISHED_FILES:
        raise ValueError('공개 파일 목록이 허용 목록과 다릅니다.')


def source_files(root):
    result=subprocess.run(['git','ls-files','--stage','-z'],cwd=root,
                          capture_output=True,check=True)
    for row in result.stdout.decode('utf-8').split('\0'):
        if not row:continue
        metadata,name=row.split('\t',1)
        mode,_,stage=metadata.split()
        if stage!='0':raise ValueError('병합 충돌이 있는 소스는 배포할 수 없습니다.')
        if mode=='160000':continue  # 웹 배포에 사용하지 않는 외부 시뮬레이터
        if mode not in ('100644','100755'):
            raise ValueError('추적 파일에 허용하지 않는 파일 형식이 있습니다.')
        path=root/name
        if path.is_symlink():raise ValueError('소스에 심볼릭 링크를 포함할 수 없습니다.')
        if ((path.name=='.env' or path.name.startswith('.env.')) and path.name!='.env.example'
                or path.suffix.lower() in {'.pem','.key','.p12','.pfx'}):
            raise ValueError('비밀 설정 또는 개인키로 의심되는 파일이 추적되고 있습니다.')
        yield name,path


def check(root,site,gitleaks):
    validate_public_files(site)
    with tempfile.TemporaryDirectory() as temporary:
        source=Path(temporary)/'source';source.mkdir()
        for name,path in source_files(root):
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
        for index,(label,path,depth) in enumerate((('소스',source,0),('배포 파일',site,2))):
            # 분할 ZIP 조각은 독립 압축 파일이 아니다. 빌드가 조각·전체 해시를
            # 검사하며, 실제 공개 gzip은 압축을 풀어 검사한다.
            report=Path(temporary)/f'report-{index}.json'
            result=subprocess.run([str(gitleaks),'dir',str(path),'--redact=100',
                f'--max-archive-depth={depth}','--no-banner','--log-level=error',
                '--report-format=json','--report-path',str(report)],capture_output=True,text=True)
            if result.returncode:
                findings=json.loads(report.read_text(encoding='utf-8')) if report.exists() else []
                for finding in findings:
                    print(json.dumps({key:finding.get(key) for key in ('RuleID','File','StartLine')},
                                     ensure_ascii=False),flush=True)
                raise ValueError(f'{label} 보안 검사 실패: 검출 {len(findings)}건, 종료 코드 {result.returncode}')
            print(f'{label} 비밀정보 검사 통과',flush=True)
    print('공개 파일 허용 목록 검사 통과',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,default=ROOT/'public')
    parser.add_argument('--gitleaks',type=Path,required=True)
    args=parser.parse_args()
    check(ROOT,args.site.resolve(),args.gitleaks.resolve())
