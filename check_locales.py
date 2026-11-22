"""List every string the app passes through t() that a locale file does not translate.

Usage: python check_locales.py [--unused]
Exits with status 1 when a locale is missing a string or a translation drops a {placeholder}.
"""
import ast
import json
import re
import string
import sys
from pathlib import Path

from procedure_support import RULES

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
LOCALES = DATA / 'locales'


def _tree(name):
    return ast.parse((BASE / name).read_text(encoding='utf-8'))


def _strings(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def literal_calls():
    """Strings written directly as t('...') in app.py, including implicitly concatenated ones."""
    return {
        node.args[0].value for node in ast.walk(_tree('app.py'))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 't'
        and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
    }


def dynamic_strings():
    """Strings that reach t() through variables: statuses, labels, weaknesses and dataset names."""
    found = set()
    for node in ast.walk(_tree('app.py')):
        if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == 'availability_notes' for target in node.targets):
            found.update(value.value for value in node.value.values)
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'show_checklist_excerpts':
            found.add(node.args[0].value)
        if isinstance(node, ast.keyword) and node.arg == 'options' and isinstance(node.value, ast.List):
            found.update(_strings(node.value))
    found.add('No specific procedure')

    for node in ast.walk(_tree('ranking.py')):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'append' \
                and getattr(node.func.value, 'id', '') == 'weaknesses':
            found.update(_strings(node))
        if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') in ('CLEANING_FLAG_EXPLANATIONS', 'flag_explanations')
                                                for target in node.targets):
            found.update(value.value for value in node.value.values)
        if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == 'weaknesses' for target in node.targets):
            found.update(_strings(node.value))

    for node in ast.walk(_tree('procedure_support.py')):
        if isinstance(node, ast.Assign) and any(isinstance(target, (ast.Name, ast.Tuple)) and 'status' in ast.unparse(target)
                                                for target in node.targets):
            found.update(value for value in _strings(node.value) if value[:1].isupper())
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == '_requirements':
            found.update(value for value in _strings(node) if value.startswith('Related '))
    for name, (_, _, equipment, capabilities, _) in RULES.items():
        found.add(name)
        found.update(label for label, _ in (*equipment, *capabilities))

    symptoms = json.loads((DATA / 'symptoms.json').read_text(encoding='utf-8'))
    for category, entries in symptoms.items():
        found.add(category)
        found.update(entries)
    procedures = json.loads((DATA / 'procedures.json').read_text(encoding='utf-8'))['procedures']
    found.update(item['name'] for item in procedures)
    found.update(('primary', 'secondary', 'tertiary'))
    return found


def readable_specialties():
    """Specialty names as app.readable_specialty() displays them, without starting Streamlit."""
    function = next(node for node in _tree('app.py').body
                    if isinstance(node, ast.FunctionDef) and node.name == 'readable_specialty')
    namespace = {'re': re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), namespace)
    readable_specialty = namespace['readable_specialty']
    symptoms = json.loads((DATA / 'symptoms.json').read_text(encoding='utf-8'))
    return {readable_specialty(code) for entries in symptoms.values() for codes in entries.values() for code in codes}


def placeholders(text):
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def main():
    sources = literal_calls() | dynamic_strings() | readable_specialties()
    failed = False
    for path in sorted(LOCALES.glob('*.json')):
        if path.name == 'languages.json':
            continue
        locale = json.loads(path.read_text(encoding='utf-8'))
        missing = sorted(sources - set(locale))
        broken = sorted(key for key in sources & set(locale) if placeholders(key) != placeholders(locale[key]))
        print(f'{path.stem}: {len(missing)} missing, {len(broken)} with mismatched placeholders')
        for text in missing:
            print(f'  missing: {text}')
        for text in broken:
            print(f'  placeholders: {text}')
        if '--unused' in sys.argv:
            for text in sorted(set(locale) - sources):
                print(f'  unused: {text}')
        failed = failed or bool(missing or broken)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
