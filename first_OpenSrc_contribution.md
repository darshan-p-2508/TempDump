I went through the repository you uploaded, including the data utilities, NDJSON converter, configuration validation, and existing test suite.

## The one change I recommend

**Fix `get_split_fraction()` so direct callers cannot pass invalid dataset fractions.**

This is a real correctness improvement, not a cosmetic contribution.

### Why this is worth contributing

Ultralytics already validates `fraction` when it comes through the main configuration system. But the lower-level public utility:

```python
get_split_fraction()
```

can currently be called directly with values such as:

```python
-0.1
1.5
float("nan")
float("inf")
True
"0.5"
```

and some of those values are accepted.

I reproduced this in your repository:

```text
-0.1  -> -0.1
nan   -> nan
inf   -> inf
True  -> 1.0
```

That means invalid dataset-subset configuration can bypass the normal config validator and eventually produce confusing downstream behavior.

Your repository already has tests for the higher-level configuration validation, but it does **not** test the lower-level helper itself.

This contribution makes the validation consistent at the utility boundary.

I also checked the current upstream implementation: the same `get_split_fraction()` logic is still present there, so you're not simply duplicating an existing upstream fix. ([GitHub][1])

---

# Exactly what you should change

You only need to modify **2 files**:

```text
ultralytics/data/utils.py
tests/test_python.py
```

No new feature, no dependencies, no documentation changes, no huge PR.

---

## 1. Change `ultralytics/data/utils.py`

Find:

```python
def get_split_fraction(fraction: float | list[float | int], split: str) -> float | int:
```

Inside it, you currently have:

```python
    fraction = float(fraction) if fraction in {0, 1} else fraction
    if split in {"train", "val"} and fraction == 0:
        raise ValueError(f"{split} fraction must select at least one image")
    return fraction
```

### Replace that section with:

```python
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError(f"fraction must be an int, float, or list, not {type(fraction).__name__}")
    if not np.isfinite(fraction) or fraction < 0 or (isinstance(fraction, float) and fraction > 1):
        raise ValueError(f"Invalid {split} fraction {fraction}. Use an integer count >1 or ratio (0, 1].")
    fraction = float(fraction) if fraction in {0, 1} else fraction
    if split in {"train", "val"} and fraction == 0:
        raise ValueError(f"{split} fraction must select at least one image")
    return fraction
```

So the complete function becomes:

```python
def get_split_fraction(fraction: float | list[float | int], split: str) -> float | int:
    """Return a split ratio/count, normalizing boundary values to 0.0 (none) or 1.0 (all).

    Args:
        fraction (float | list[float | int]): Dataset fraction (ratio or image count), or a per-split list ordered
            as [train, val, test]. A scalar only applies to the train split; missing list entries default to 1.0.
        split (str): Dataset split name, e.g. 'train', 'val', or 'test'.

    Returns:
        (float | int): Fraction of the split to use as a ratio (float) or image count (int).

    Raises:
        ValueError: If the resolved fraction is 0 for the 'train' or 'val' split.
    """
    if isinstance(fraction, list) and split in (splits := ("train", "val", "test")):
        index = splits.index(split)
        fraction = fraction[index] if index < len(fraction) else 1.0
    elif split != "train":
        fraction = 1.0

    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError(f"fraction must be an int, float, or list, not {type(fraction).__name__}")
    if not np.isfinite(fraction) or fraction < 0 or (isinstance(fraction, float) and fraction > 1):
        raise ValueError(f"Invalid {split} fraction {fraction}. Use an integer count >1 or ratio (0, 1].")

    fraction = float(fraction) if fraction in {0, 1} else fraction

    if split in {"train", "val"} and fraction == 0:
        raise ValueError(f"{split} fraction must select at least one image")

    return fraction
```

---

# 2. Add regression tests

Open:

```text
tests/test_python.py
```

You already have this section around the existing fraction tests:

```python
    with pytest.raises(TypeError, match="fraction"):
        get_cfg(overrides={"fraction": True})
    assert get_cfg(overrides={"auto_augment": None}).auto_augment is None
```

Immediately after the `get_cfg(...fraction=True)` test, add:

```python
    for value in (-0.1, 1.5, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises((TypeError, ValueError), match="fraction"):
            get_split_fraction(value, "train")
    assert get_split_fraction(2, "train") == 2
    assert get_split_fraction(1.0, "train") == 1.0
```

So that section should look like:

```python
    assert type(get_split_fraction([1, 1, 0], "train")) is float
    assert type(get_split_fraction([1, 1, 0], "test")) is float
    with pytest.raises(ValueError, match="val fraction"):
        get_split_fraction([1, 0], "val")
    with pytest.raises(TypeError, match="fraction"):
        get_cfg(overrides={"fraction": True})

    for value in (-0.1, 1.5, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises((TypeError, ValueError), match="fraction"):
            get_split_fraction(value, "train")
    assert get_split_fraction(2, "train") == 2
    assert get_split_fraction(1.0, "train") == 1.0

    assert get_cfg(overrides={"auto_augment": None}).auto_augment is None
```

That's it.

---

# 3. Set up the repository properly

Since your downloaded ZIP doesn't contain the Git history, **don't initialize the ZIP as a new Git repository**.

You want the real fork → branch → PR workflow.

First fork:

[Ultralytics repository](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com)

Click:

**Fork → Create fork**

Then on Windows PowerShell:

```powershell
cd C:\Users\YOUR_NAME\Documents

git clone https://github.com/YOUR_USERNAME/ultralytics.git

cd ultralytics

git remote add upstream https://github.com/ultralytics/ultralytics.git

git remote -v
```

You should see:

```text
origin    https://github.com/YOUR_USERNAME/ultralytics.git
upstream  https://github.com/ultralytics/ultralytics.git
```

GitHub's documented open-source workflow is exactly this fork → branch → change → test → push → pull-request process. ([GitHub Docs][2])

---

# 4. Make sure your fork is current

Before creating your branch:

```powershell
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

Then create a dedicated branch:

```powershell
git checkout -b fix-fraction-validation
```

---

# 5. Open it in VS Code

```powershell
code .
```

Make the two changes above.

Your final modified files should be exactly:

```text
ultralytics/data/utils.py
tests/test_python.py
```

Check:

```powershell
git status
```

You should see only those two files modified.

---

# 6. Install development dependencies

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

# 7. Run the exact regression test

This is the important test:

```powershell
python -m pytest tests/test_python.py::test_cfg_rejects_fuzzed_values -q
```

I ran this exact test against your uploaded repository after making the proposed change.

Result:

```text
1 passed in 0.45s
```

So the actual change has been verified.

---

# 8. Run the broader Python test file

Then run:

```powershell
python -m pytest tests/test_python.py -q
```

Some Ultralytics tests can be environment-dependent, especially tests involving CUDA, external assets, or optional integrations. Don't panic if an unrelated environment-specific test is skipped or fails.

What matters for your PR is that the changed test and relevant test suite pass, and that you don't introduce unrelated failures.

---

# 9. Check formatting/linting

Run:

```powershell
ruff check ultralytics/data/utils.py tests/test_python.py
```

Then:

```powershell
ruff format --check ultralytics/data/utils.py tests/test_python.py
```

If Ruff reports that formatting needs to be applied:

```powershell
ruff format ultralytics/data/utils.py tests/test_python.py
```

Then run the tests again.

---

# 10. Inspect your actual diff

This is extremely important before committing:

```powershell
git diff --check
```

Then:

```powershell
git diff
```

Your diff should essentially contain:

### `ultralytics/data/utils.py`

```diff
+    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
+        raise TypeError(...)
+    if not np.isfinite(fraction) or fraction < 0 or (...):
+        raise ValueError(...)
```

### `tests/test_python.py`

```diff
+    for value in (-0.1, 1.5, float("nan"), float("inf"), True, "0.5"):
+        with pytest.raises((TypeError, ValueError), match="fraction"):
+            get_split_fraction(value, "train")
+    assert get_split_fraction(2, "train") == 2
+    assert get_split_fraction(1.0, "train") == 1.0
```

**Nothing else.**

This is exactly the kind of focused PR Ultralytics' contribution guidelines encourage; their guidelines specifically say first-time contributors should keep changes small and focused, with tests for new behavior. ([GitHub Docs][3])

---

# 11. Commit it

Use a meaningful commit message:

```powershell
git add ultralytics/data/utils.py tests/test_python.py
```

Then:

```powershell
git commit -m "Validate dataset fraction inputs"
```

Then:

```powershell
git push -u origin fix-fraction-validation
```

---

# 12. Create the Pull Request

Go to your fork on GitHub.

You should get:

> Compare & pull request

Select:

```text
base repository: ultralytics/ultralytics
base: main

head repository: YOUR_USERNAME/ultralytics
compare: fix-fraction-validation
```

Do **not** accidentally make the PR target your own fork.

### PR title

Use:

```text
Validate dataset fraction inputs
```

### PR description

## What does this PR do?

Validates dataset `fraction` values inside `get_split_fraction()` so direct callers cannot pass invalid values through to dataset selection.

## Why is this needed?

The main configuration validation already rejects invalid `fraction` values, but callers using `get_split_fraction()` directly could bypass that validation.

Values such as negative fractions, non-finite floats, booleans, and floats greater than 1 could previously be accepted by the helper and fail later in dataset processing with less useful errors.

This change makes the lower-level utility enforce the same fraction contract as the configuration layer.

## Changes

* Reject boolean and non-numeric fraction values.
* Reject negative and non-finite fractions.
* Reject floating-point values greater than 1.
* Preserve integer values greater than 1 as explicit image counts.
* Preserve existing `0`/`1` normalization and train/validation zero checks.
* Add regression coverage for invalid direct calls.

## Testing

* `python -m pytest tests/test_python.py::test_cfg_rejects_fuzzed_values -q`
* `ruff check ultralytics/data/utils.py tests/test_python.py`
* `ruff format --check ultralytics/data/utils.py tests/test_python.py`

Keep the PR focused. Don't add screenshots, unrelated refactors, README changes, or extra features.

---

# Why I picked this instead of an easier "achievement hack"

I specifically **wouldn't** recommend something like:

* changing the README
* adding a random example
* fixing whitespace
* adding a trivial comment
* adding a meaningless test
* changing badges
* adding a personal file
* making a cosmetic UI/documentation change

Those might produce GitHub activity, but they don't make a convincing open-source contribution.

This one has a clear engineering story:

**configuration validation → lower-level utility → malformed input → obscure downstream behavior → fail early with a useful error → regression tests.**

It is also only a **two-file, small-scope PR**, which fits Ultralytics' stated expectations for first-time contributors. ([GitHub Docs][3])

And because the current upstream helper still has this validation gap, you're proposing an actual improvement rather than copying an already-merged change. ([GitHub][1])

### One important expectation about the achievement

The PR itself gives you a legitimate GitHub contribution event, but **don't treat the achievement badge as guaranteed**. GitHub says achievements are based on qualifying GitHub events and that the achievement system is currently subject to change. ([GitHub Docs][4])

The meaningful goal should therefore be:

**make the valid contribution → push it → open the PR → let CI run → respond to review → ideally get it merged.**

That gives you both a genuine contribution to Ultralytics and the GitHub activity you're looking for.

[1]: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py?utm_source=chatgpt.com "ultralytics/ultralytics/data/utils.py at main · ultralytics/ultralytics · GitHub"
[2]: https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project?tool=cli&utm_source=chatgpt.com "Contributing to a project - GitHub Docs"
[3]: https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-open-source?utm_source=chatgpt.com "Contributing to open source - GitHub Docs"
[4]: https://docs.github.com/en/account-and-profile/reference/profile-reference?utm_source=chatgpt.com "Profile reference - GitHub Docs"
