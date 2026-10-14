from __future__ import annotations
import copy
import json
from typing import Any
from neos.workflow.toolsets import DISABLED_TOOL_NAMES, adapt_function_description
NEOS_ARRAY_TOOL_SCHEMA = 'native_array'
NEOS_SCHEMA_VERSIONS = (NEOS_ARRAY_TOOL_SCHEMA,)
DEFAULT_NEOS_SCHEMA_VERSION = NEOS_ARRAY_TOOL_SCHEMA

class ArgumentNormalizationError(ValueError):
    pass
WEB_VISIT_MAX_PAGES = 5
IMAGE_SEARCH_MAX_QUERIES = 5
REVERSE_IMAGE_SEARCH_MAX_INPUTS = 3
FETCH_IMAGE_MAX_URLS = 5
CROP_IMAGE_MAX_PER_IMAGE = 5
CROP_IMAGE_MAX_TOTAL = 10
TEXT_SEARCH_FUNCTION = {'name': 'TextSearch', 'description': 'Search the live web. Each query returns up to 10 lightweight title-URL-snippet results; it does not load page or image pixels.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'description': 'A non-empty array of focused web queries.'}}, 'required': ['query'], 'additionalProperties': False}}
WEB_VISIT_FUNCTION = {'name': 'WebVisit', 'description': 'Visit up to five webpages and return goal-conditioned textual evidence plus lightweight page-image references. Referenced images are not visually observed until FetchImage loads them.', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 5, 'description': 'A non-empty array of at most 5 complete http(s) URLs.'}, 'goal': {'type': 'string', 'description': 'The non-empty evidence goal for these pages.'}}, 'required': ['url', 'goal'], 'additionalProperties': False}}
IMAGE_SEARCH_FUNCTION = {'name': 'ImageSearch', 'description': 'Search for online images with up to five text queries. Return at most five goal-conditioned candidate references per query; candidate pixels are not placed in the policy context. Use FetchImage on a selected direct image URL before making a visual claim.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 5, 'description': 'A non-empty array of at most 5 image queries.'}, 'goal': {'type': 'string', 'description': 'The visual evidence goal used to rank candidates.'}}, 'required': ['query', 'goal'], 'additionalProperties': False}}
REVERSE_IMAGE_SEARCH_FUNCTION = {'name': 'ReverseImageSearch', 'description': 'Run reverse-image search on up to three loaded image tokens and/or public image URLs. Return goal-conditioned textual matches and lightweight candidate references, never implicit candidate pixels.', 'parameters': {'type': 'object', 'properties': {'image_token': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 3, 'description': 'A non-empty array of at most 3 loaded img_N tokens.'}, 'image_url': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 3, 'description': 'A non-empty array of at most 3 public http(s) image URLs.'}, 'goal': {'type': 'string', 'description': 'The identity or visual-match goal.'}}, 'required': ['goal'], 'anyOf': [{'required': ['image_token']}, {'required': ['image_url']}], 'additionalProperties': False}}
FETCH_IMAGE_FUNCTION = {'name': 'FetchImage', 'description': 'Load up to five selected image URLs in parallel. Successfully decoded pixels are appended to the visual register as new img_N tokens in request order. Every successful result record directly includes its new output_ref.', 'parameters': {'type': 'object', 'properties': {'urls': {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 5, 'description': 'A non-empty ordered array of at most 5 direct image URL strings.'}}, 'required': ['urls'], 'additionalProperties': False}}
CROP_IMAGE_FUNCTION = {'name': 'CropImage', 'description': 'Crop loaded visual observations in parallel. crop_config maps each img_N token to one normalized [left, top, right, bottom] box in the 0..1000 coordinate space or a list of boxes. At most five boxes per image and ten total are materialized as reusable new img_N observations in deterministic request order. Every successful crop record directly includes its new output_ref.', 'parameters': {'type': 'object', 'properties': {'crop_config': {'type': 'object', 'minProperties': 1, 'additionalProperties': {'anyOf': [{'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4}, {'type': 'array', 'minItems': 1, 'maxItems': 5, 'items': {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4}}]}, 'description': 'Mapping from loaded img_N tokens to normalized 0..1000 boxes.'}}, 'required': ['crop_config'], 'additionalProperties': False}}
NEOS_FUNCTIONS = [TEXT_SEARCH_FUNCTION, WEB_VISIT_FUNCTION, IMAGE_SEARCH_FUNCTION, REVERSE_IMAGE_SEARCH_FUNCTION, FETCH_IMAGE_FUNCTION, CROP_IMAGE_FUNCTION]
if DISABLED_TOOL_NAMES:
    NEOS_FUNCTIONS = [f for f in NEOS_FUNCTIONS if f['name'] not in DISABLED_TOOL_NAMES]
    for function in NEOS_FUNCTIONS:
        adapt_function_description(function)
NEOS_TOOL_NAMES = frozenset((function['name'] for function in NEOS_FUNCTIONS))
NEOS_FUNCTION_BY_NAME = {function['name']: function for function in NEOS_FUNCTIONS}

def get_neos_functions(schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> list[dict]:
    if schema_version != DEFAULT_NEOS_SCHEMA_VERSION:
        raise ValueError(f'unsupported Neos tool schema: {schema_version!r}')
    return copy.deepcopy(NEOS_FUNCTIONS)

def get_neos_function_by_name(schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> dict[str, dict]:
    return {function['name']: function for function in get_neos_functions(schema_version)}

def _json_type_name(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    if isinstance(value, (int, float)):
        return 'number'
    return type(value).__name__

def _container_branches(schema: dict) -> list[dict]:
    branches = []
    if schema.get('type') in {'array', 'object'}:
        branches.append(schema)
    for keyword in ('anyOf', 'oneOf'):
        for branch in schema.get(keyword, []):
            if isinstance(branch, dict) and branch.get('type') in {'array', 'object'}:
                branches.append(branch)
    unique = []
    seen = set()
    for branch in branches:
        marker = json.dumps(branch, sort_keys=True, separators=(',', ':'))
        if marker not in seen:
            seen.add(marker)
            unique.append(branch)
    return unique

def _complete_json_container_kind(value: str) -> str | None:
    stripped = value.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        return 'array'
    if stripped.startswith('{') and stripped.endswith('}'):
        return 'object'
    return None

def _normalize_schema_value(value: Any, schema: dict, *, tool_name: str, path: str, events: list[dict[str, str]]) -> Any:
    pass
    from jsonschema import Draft202012Validator
    container_branches = _container_branches(schema)
    if isinstance(value, str) and container_branches:
        container_kind = _complete_json_container_kind(value)
        if container_kind is not None:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ArgumentNormalizationError(f'{tool_name} {path}: complete JSON {container_kind} string is malformed: {exc.msg}') from exc
            if _json_type_name(decoded) != container_kind:
                raise ArgumentNormalizationError(f'{tool_name} {path}: decoded container type changed unexpectedly')
            candidates = [branch for branch in container_branches if branch.get('type') == container_kind and Draft202012Validator(branch).is_valid(decoded)]
            if not candidates:
                raise ArgumentNormalizationError(f'{tool_name} {path}: decoded {container_kind} does not match any container branch in the schema')
            if len(candidates) > 1:
                raise ArgumentNormalizationError(f'{tool_name} {path}: decoded {container_kind} ambiguously matches {len(candidates)} container branches')
            normalized = _normalize_schema_value(decoded, candidates[0], tool_name=tool_name, path=path, events=events)
            events.append({'tool': tool_name, 'path': path, 'from_type': 'string', 'to_type': container_kind, 'reason': f'json_container_string_matches_{container_kind}_schema'})
            return normalized
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        properties = schema.get('properties') or {}
        additional = schema.get('additionalProperties')
        for key, item in list(result.items()):
            child_schema = properties.get(key)
            if child_schema is None and isinstance(additional, dict):
                child_schema = additional
            if isinstance(child_schema, dict):
                result[key] = _normalize_schema_value(item, child_schema, tool_name=tool_name, path=f'{path}.{key}', events=events)
        return result
    if isinstance(value, list):
        item_schema = schema.get('items')
        if not isinstance(item_schema, dict):
            matching = [branch for branch in container_branches if branch.get('type') == 'array']
            if len(matching) == 1:
                item_schema = matching[0].get('items')
        if isinstance(item_schema, dict):
            return [_normalize_schema_value(item, item_schema, tool_name=tool_name, path=f'{path}[{index}]', events=events) for index, item in enumerate(value)]
    return copy.deepcopy(value)

def normalize_tool_arguments(tool_name: str, arguments: object, *, schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> tuple[dict, list[dict[str, str]]]:
    pass
    function = get_neos_function_by_name(schema_version).get(tool_name)
    if function is None:
        raise ArgumentNormalizationError(f'unknown tool {tool_name!r}')
    if not isinstance(arguments, dict):
        raise ArgumentNormalizationError('arguments must be a JSON object')
    events: list[dict[str, str]] = []
    normalized = _normalize_schema_value(arguments, function['parameters'], tool_name=tool_name, path='$', events=events)
    error = validate_tool_arguments(tool_name, normalized, schema_version=schema_version)
    if error is not None:
        raise ArgumentNormalizationError(f'invalid {tool_name} arguments: {error}')
    return (normalized, events)

def validate_tool_arguments(name: str, arguments: object, *, schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> str | None:
    pass
    function = get_neos_function_by_name(schema_version).get(name)
    if function is None:
        return f'unknown tool {name!r}'
    if not isinstance(arguments, dict):
        return 'arguments must be a JSON object'
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(function['parameters']).iter_errors(arguments), key=lambda error: (tuple((str(part) for part in error.absolute_path)), error.message))
    if not errors and name == 'ReverseImageSearch':
        total_inputs = sum((len(value) if isinstance(value, list) else 1 for key in ('image_token', 'image_url') if (value := arguments.get(key)) is not None))
        if total_inputs > REVERSE_IMAGE_SEARCH_MAX_INPUTS:
            return f'image_token and image_url contain {total_inputs} inputs in total; maximum is {REVERSE_IMAGE_SEARCH_MAX_INPUTS}'
    if not errors and name == 'CropImage':
        total_boxes = 0
        for boxes in arguments['crop_config'].values():
            if isinstance(boxes, list) and len(boxes) == 4 and all((isinstance(value, (int, float)) for value in boxes)):
                normalized_boxes = [boxes]
            else:
                normalized_boxes = boxes
            total_boxes += len(normalized_boxes)
            for left, top, right, bottom in normalized_boxes:
                if not left < right or not top < bottom:
                    return 'crop_config boxes must satisfy left < right and top < bottom'
        if total_boxes > CROP_IMAGE_MAX_TOTAL:
            return f'crop_config contains {total_boxes} boxes in total; maximum is {CROP_IMAGE_MAX_TOTAL}'
    if not errors:
        return None
    rendered = []
    for error in errors[:5]:
        path = '.'.join((str(part) for part in error.absolute_path)) or '$'
        rendered.append(f'{path}: {error.message}')
    if len(errors) > 5:
        rendered.append(f'... and {len(errors) - 5} more schema errors')
    return '; '.join(rendered)

def openai_tool_schema(function: dict) -> dict:
    return {'type': 'function', 'function': copy.deepcopy(function)}

def verl_openai_tool_schema(function: dict) -> dict:
    pass
    visible = copy.deepcopy(function)
    rendered_properties = {}
    for name, schema in visible['parameters'].get('properties', {}).items():
        if 'type' in schema:
            property_type = schema['type']
        else:
            property_type = [option['type'] for option in schema.get('anyOf', []) if isinstance(option, dict) and 'type' in option]
        rendered = {'type': property_type, 'description': schema.get('description')}
        if 'enum' in schema:
            rendered['enum'] = schema['enum']
        rendered_properties[name] = rendered
    visible['parameters'] = {'type': 'object', 'properties': rendered_properties, 'required': visible['parameters'].get('required', [])}
    return {'type': 'function', 'function': visible}

def get_neos_tool_schemas(schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> list[dict]:
    return [openai_tool_schema(function) for function in get_neos_functions(schema_version)]

def all_openai_tool_schemas(schema_version: str=DEFAULT_NEOS_SCHEMA_VERSION) -> list[dict]:
    pass
    return get_neos_tool_schemas(schema_version)
