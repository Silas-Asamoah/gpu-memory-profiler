# Release Checklist

This checklist ensures a smooth release process for GPU Memory Profiler.

## Pre-Release Preparation

### 1. Code Quality

-   [ ] All tests pass: `python3 -m pytest`
-   [ ] Code coverage is above 80%: `python3 -m pytest --cov=gpumemprof --cov=tfmemprof`
-   [ ] Code is formatted: `python3 -m black .`
-   [ ] Imports are sorted: `python3 -m isort .`
-   [ ] Linting passes: `python3 -m flake8`
-   [ ] Type checking passes: `python3 -m mypy gpumemprof tfmemprof`
-   [ ] TUI launches successfully: `gpu-profiler`

### 2. Documentation

-   [ ] README.md is up to date
-   [ ] All documentation in `/docs/` is current
-   [ ] API documentation is complete
-   [ ] Examples are working and documented
-   [ ] Installation instructions are clear

### 3. Version Management

-   [ ] Version is updated in `pyproject.toml` (if not using dynamic versioning)
-   [ ] CHANGELOG.md is updated with new features and fixes
-   [ ] Version is consistent across all files

## Release Process

### 1. Create Release Branch

```bash
git checkout -b release/v0.1.0
git push origin release/v0.1.0
```

### 2. Final Testing

```bash
# Clean environment
python3 -m venv test_env
source test_env/bin/activate

# Install from source
pip install -e .

# Run all tests
python3 -m pytest

# Test CLI tools
gpumemprof --help
tfmemprof --help

# Test installation
pip install -e .[test]
python3 -m pytest tests/
```

### 3. Build Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build package
python3 -m build

# Verify build
python3 -m twine check dist/*
```

### 4. Test Package Installation

```bash
# Test installation from built package
pip install dist/gpu_memory_profiler-*.whl

# Verify installation
python3 -c "import gpumemprof; print(gpumemprof.__version__)"
python3 -c "import tfmemprof; print(tfmemprof.__version__)"

# Test CLI tools
gpumemprof --version
tfmemprof --version
```

## Publishing

### 1. Test PyPI (Optional)

```bash
# Upload to Test PyPI first
python3 -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ gpu-memory-profiler
```

### 2. PyPI Release

```bash
# Upload to PyPI
python3 -m twine upload dist/*

# Verify upload
python3 -m pip install gpu-memory-profiler
```

### 3. GitHub Release

-   [ ] Create release tag: `git tag v0.1.0`
-   [ ] Push tag: `git push origin v0.1.0`
-   [ ] Create GitHub release with release notes
-   [ ] Upload built packages to GitHub release

## Post-Release

### 1. Update Documentation

-   [ ] Update PyPI badges in README.md
-   [ ] Update installation instructions
-   [ ] Update version references

### 2. Announce Release

-   [ ] Update project status
-   [ ] Share on social media
-   [ ] Notify contributors
-   [ ] Update any external references

### 3. Prepare Next Release

-   [ ] Create new development branch
-   [ ] Update version to next development version
-   [ ] Update CHANGELOG.md with "Unreleased" section

## Automated Release (GitHub Actions)

### 1. Trigger Release

-   [ ] Push tag to trigger automated release
-   [ ] Verify GitHub Actions workflow runs successfully
-   [ ] Check PyPI upload in Actions logs

### 2. Verify Automated Release

-   [ ] Check PyPI for new version
-   [ ] Test installation: `pip install gpu-memory-profiler==0.1.0`
-   [ ] Verify CLI tools work
-   [ ] Run smoke tests

## Rollback Plan

If issues are discovered after release:

1. **Immediate Actions**

    - [ ] Mark release as deprecated on PyPI
    - [ ] Create hotfix branch
    - [ ] Fix critical issues

2. **Hotfix Release**

    - [ ] Increment patch version
    - [ ] Apply fixes
    - [ ] Follow release process for hotfix

3. **Communication**
    - [ ] Update GitHub release notes
    - [ ] Notify users of issues
    - [ ] Provide workarounds if needed

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):

-   **MAJOR**: Breaking changes
-   **MINOR**: New features, backward compatible
-   **PATCH**: Bug fixes, backward compatible

Examples:

-   `0.1.0` - Initial release
-   `0.1.1` - Bug fix release
-   `0.2.0` - New features
-   `1.0.0` - First stable release

## Checklist Template

Copy this template for each release:

```markdown
# Release v0.1.0 Checklist

## Pre-Release

-   [ ] Code quality checks pass
-   [ ] Documentation updated
-   [ ] Version management complete
-   [ ] Final testing completed

## Release

-   [ ] Release branch created
-   [ ] Package built successfully
-   [ ] PyPI upload completed
-   [ ] GitHub release created

## Post-Release

-   [ ] Documentation updated
-   [ ] Release announced
-   [ ] Next release prepared

## Notes

-   Any issues encountered:
-   Lessons learned:
-   Improvements for next release:
```

## Emergency Contacts

-   **Release Manager**: [Your Name]
-   **Backup**: [Backup Name]
-   **PyPI Admin**: [PyPI Username]

## Resources

-   [PyPI Upload Guide](https://packaging.python.org/tutorials/packaging-projects/)
-   [GitHub Releases](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository)
-   [Semantic Versioning](https://semver.org/)
