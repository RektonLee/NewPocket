# Figures Directory

This directory stores generated figures and visualizations from the project.

## Structure

```
figures/
├── archived/     # Historical figures and old plots
└── (current)     # Current figures (gitignored by default)
```

## Usage

- All `.png`, `.jpg`, `.svg`, and `.pdf` files in this directory are **gitignored by default**
- Only add figures to git if they are essential for documentation
- Use `archived/` for figures from past experiments that you want to keep for reference

## Adding Figures to Git

If you need to commit specific figures (e.g., for paper or documentation):

```bash
# Whitelist specific files in .gitignore
!figures/important_plot.png

# Or force add them
git add -f figures/important_plot.png
```

## Best Practices

1. **Name files descriptively**: `kcat_prediction_scatter_exp001.png` instead of `plot1.png`
2. **Use subdirectories**: Create subdirectories for different experiments or paper sections
3. **Archive old figures**: Move outdated figures to `archived/` to keep root clean
4. **Generate programmatically**: All figures should be reproducible from scripts

## Paper Figures

For paper submission figures, use `paperwriting/figures/` instead to keep them separate and version-controlled.
