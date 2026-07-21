# Contributing to RoadEye-UK

First off, thank you for considering contributing to RoadEye-UK! Whether you're fixing a bug, optimizing CV pipelines, or adding a roadmap feature, contributions are welcome.

---

## 🛠️ Local Setup

**Fork & Clone the repository:**
```bash
git clone [https://github.com/reory/roadeye-uk.git](https://github.com/reory/roadeye-uk.git)
cd roadeye-uk
```

### Set up virtual environment & dependencies:
```Bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## 📐 Development Guidelines
Code Style & Quality
Use ruff for fast linting and code formatting:

```Bash
uv run ruff check .
uv run ruff format .
```
### Testing Requirement
All new features, bug fixes, or modifications to src/ must include corresponding tests in tests/:

Run tests locally before opening a pull request:

```Bash
uv run pytest
```
Ensure all tests pass and coverage does not drop below 85%.

---

## 🚀 Submitting a Pull Request
Create a feature branch off main:

```Bash
git checkout -b feature/your-feature-name
```
- Keep commits atomic, descriptive, and clean.

- Push your branch to your fork and open a Pull Request targeting main.

- Fill out the PR description with:

- What changed and why.

- Screenshots/GIFs or test logs verifying the changes (especially for HUD or video UI updates).

---

### 💬 Questions or Bug Reports?
Feel free to open an Issue on GitHub describing the bug or feature request with steps to reproduce!