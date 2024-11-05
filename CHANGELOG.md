# Changelog

## [0.1.0] - 2024-03-05

### Added
- Flexible pattern matching in CrudControls using both regex and glob patterns
- Support for wildcard specifications (e.g., "get_thing*") to match multiple related functions
- Automatic detection of regex vs glob patterns
- Shell-style wildcards support (`*`, `?`, `[seq]`, `[!seq]`)

### Changed
- Updated CrudControls pattern matching logic to be more precise and flexible
- Improved function name matching to better handle related functions (e.g., get_thing vs get_thing_by_id)
- Fixed invalid escape sequence warning in regex character detection

### Technical Details
- Added `_is_regex_pattern()` method to detect regex patterns
- Enhanced `compile_patterns()` to handle both regex and glob patterns
- Updated `matches_pattern()` to support dual pattern matching system
- Added comprehensive pattern validation and error handling

## [0.0.2] - Initial Release

- Basic CRUD operations support
- Simple string matching for function names
- AWS SDK integration example
- Initial LangChain integration