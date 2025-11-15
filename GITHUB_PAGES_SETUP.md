# GitHub Pages Setup Instructions

This repository is configured to automatically publish Marimo notebooks as static HTML pages using GitHub Actions and GitHub Pages.

## Prerequisites

You need to enable GitHub Pages for this repository. Follow these steps:

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (gear icon in the top menu)
3. In the left sidebar, click on **Pages** (under "Code and automation")
4. Under **Source**, select **GitHub Actions** (not "Deploy from a branch")
5. Click **Save**

### 2. Workflow Triggers

The workflow automatically runs when:
- Code is pushed to the `main` or `master` branch
- Manually triggered via the **Actions** tab (using "workflow_dispatch")

### 3. First Deployment

After enabling GitHub Pages and pushing this workflow configuration:

1. The workflow will run automatically on the next push to `main`/`master`
2. Or trigger it manually:
   - Go to the **Actions** tab
   - Click on "Publish Marimo Notebooks to GitHub Pages"
   - Click "Run workflow"
   - Select the branch and click "Run workflow"

### 4. Accessing the Published Notebooks

Once deployed, your notebooks will be available at:

```
https://<username>.github.io/<repository-name>/
```

For example, if your GitHub username is `mojzis` and repository is `volby_check`:
```
https://mojzis.github.io/volby_check/
```

## What Gets Published

The workflow exports and publishes these Marimo notebooks:

- **suspicious_zeros_with_obec_analysis.py** → Enhanced analysis (recommended)
- **suspicious_zeros_analysis.py** → Basic probability analysis
- **suspicious_cases_analysis.py** → Additional suspicious cases
- **suspicious_zeros_obec_specific.py** → Municipality-specific analysis (if exists)

Plus an **index.html** landing page that lists all available notebooks.

## Workflow Details

The GitHub Actions workflow (`.github/workflows/publish-notebooks.yml`) performs these steps:

1. **Checks out the code** from the repository
2. **Sets up Python 3.11** and the `uv` package manager
3. **Installs dependencies** using `uv sync`
4. **Exports each Marimo notebook to HTML** using `marimo export html`
5. **Creates an index page** with links to all notebooks
6. **Deploys to GitHub Pages** using the official GitHub Pages action

## Troubleshooting

### Workflow fails with permission error

Make sure GitHub Pages source is set to "GitHub Actions" (not "Deploy from a branch").

### 404 error when accessing the site

- Wait a few minutes after the first deployment
- Check that the workflow completed successfully in the Actions tab
- Verify the Pages URL in Settings → Pages

### Notebooks not updating

- The workflow only runs on pushes to `main`/`master` branch
- Check if your changes were merged to the main branch
- You can manually trigger the workflow from the Actions tab

## Local Testing

To test notebook exports locally before pushing:

```bash
# Install dependencies
uv sync

# Export a single notebook
uv run marimo export html suspicious_zeros_with_obec_analysis.py -o test_output.html

# Open in browser to verify
```

## Updating the Workflow

The workflow configuration is in `.github/workflows/publish-notebooks.yml`. You can modify it to:

- Add more notebooks to export
- Change the Python version
- Customize the index page styling
- Add additional build steps

After modifying, commit and push to trigger a new deployment.
