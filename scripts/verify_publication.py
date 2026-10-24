"""Check public files, local links, private-data patterns, and reachable Git trees."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
CHECKS = ['publication-checks.json', 'docs/publication-checks.json']
ALLOWED_ROOT = {'.gitignore', '.gitattributes', 'README.md', 'NOTICE.md', 'PRIVACY.md', 'publication-checks.json'}
PATTERNS = {
    'personal_windows_path': re.compile(r'(?i)[a-z]:[\\/](?:users|documents and settings)[\\/]'),
    'personal_unix_path': re.compile(r'(?i)/(?:users|home)/[a-z0-9_.-]+/'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'api_secret': re.compile(r'\bsk-[A-Za-z0-9_-]{24,}\b'),
    'private_ipv4': re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|127\.0\.0\.1)\b'),
}
FORBIDDEN_PATHS = re.compile(r'(?i)(?:^|/)(?:learner-state|diagnostic|handoff|\.env|__pycache__|originals|publisher-text|papers|portable_course|text)(?:[./_-]|$)|\.(?:png|jpe?g|pdf|zip|log)$')


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = []; self.remote = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ['href', 'src']:
            if attrs.get(key):
                self.links.append(attrs[key])
                if tag in ['script', 'img', 'link', 'iframe'] and urlsplit(attrs[key]).scheme in ['https', 'http']:
                    self.remote.append(attrs[key])


def git(*arguments, binary=False):
    result = subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode('utf-8', errors='replace')


def allowed(path):
    return path in ALLOWED_ROOT or path.startswith(('docs/', 'scripts/'))


def scan(path, payload, errors, scope):
    if not allowed(path) or FORBIDDEN_PATHS.search(path):
        errors.append(f'{scope}: unexpected path {path}')
    # Torch state dictionaries are generated teaching assets, never personal data.
    if path.endswith('.pt'):
        assert Path(path).name in ['vision-weights.pt', 'routing-weights.pt']
        text = payload.decode('utf-8', errors='ignore')
    else:
        text = payload.decode('utf-8', errors='replace')
    for name, regex in PATTERNS.items():
        if regex.search(text):
            errors.append(f'{scope}: {name} in {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    errors, inspections = [], {}
    files = [p for p in DOCS.rglob('*') if p.is_file()]
    files += [ROOT / n for n in ['.gitignore', '.gitattributes', 'README.md', 'NOTICE.md', 'PRIVACY.md']]
    files += [p for p in (ROOT / 'scripts').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for path in files:
        scan(path.relative_to(ROOT).as_posix(), path.read_bytes(), errors, 'working')
        if path.suffix == '.html':
            inspector = Inspector(); inspector.feed(path.read_text(encoding='utf-8'))
            inspections[path.resolve()] = inspector
            if len(inspector.ids) != len(set(inspector.ids)):
                errors.append('Duplicate HTML IDs: ' + path.relative_to(ROOT).as_posix())
            if inspector.remote:
                errors.append('Remote dependency: ' + path.relative_to(ROOT).as_posix())
    links = 0
    for path, inspector in inspections.items():
        for href in inspector.links:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(DOCS):
                errors.append(f'Link escapes Pages root: {path.name} -> {href}'); continue
            if target.name == 'publication-checks.json':
                # Written by this validator after all checks finish.
                continue
            if not target.is_file():
                errors.append(f'Broken local link: {path.name} -> {href}'); continue
            if parsed.fragment and target.suffix == '.html':
                if unquote(parsed.fragment) not in inspections[target].ids:
                    errors.append(f'Broken anchor: {path.name} -> {href}')
            links += 1
    commits = 0
    checked_blobs = set()
    if args.history:
        revisions = git('rev-list', '--all').splitlines()
        commits = len(revisions)
        for rev in revisions:
            for row in git('ls-tree', '-r', rev).splitlines():
                metadata, path = row.split('\t', 1)
                blob = metadata.split()[2]
                if (blob, path) not in checked_blobs:
                    scan(path, git('cat-file', '-p', blob, binary=True), errors, 'history')
                    checked_blobs.add((blob, path))
            identity = git('show', '-s', '--format=%an%n%cn%n%B', rev)
            if not identity.startswith('az9713\naz9713\n'):
                errors.append('Unexpected public author identity: ' + rev)
        # Validate the exact staged snapshot as well as working files.
        for path in git('ls-files').splitlines():
            scan(path, git('show', ':' + path, binary=True), errors, 'index')
    result = {'passed': not errors, 'html_pages': len(inspections),
              'local_links_checked': links, 'public_files_scanned': len(files),
              'reachable_commits_checked': commits, 'unique_history_entries_scanned': len(checked_blobs),
              'email_policy': 'allowed', 'original_source_documents': 'linked_at_publishers',
              'errors': errors}
    for name in CHECKS:
        (ROOT / name).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
