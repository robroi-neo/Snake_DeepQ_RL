# Snake Game Using Deep Q-Learning

## Summary
This project is currently a direct implementation based on a tutorial previously followed.  
— Robroi

## Objectives
Enhance or extend the project beyond the scope of the original tutorial by introducing additional features and improvements.

### Planned Features
- Save each new training milestone (e.g., when the model reaches a new high score).
- Potential gameplay modifications:
  - Add walls or obstacles.
  - Introduce alternative food types (e.g., larger items with higher score rewards for faster consumption).
- Improve the graphical user interface (GUI) for a more polished presentation.
- Package the project as a standalone desktop application.

---

## Developer Notes

### Environment Setup (Conda)

**to install env**
- This is for installing the snake_rl_env
```bash
conda env create -f snake_rl_env.yml
```

**Activate the environment**
```bash
conda activate pygame_env
```

**to generate conda environment**
- this will create a blank environment no need to use this
```bash
conda env export > snake_rl_env.yml
```