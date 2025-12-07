# Snake Game Using Deep Q-Learning

## Summary
This project is currently a direct implementation based on a tutorial previously followed.  
— Robroi

## Objectives
Enhance or extend the project beyond the scope of the original tutorial by introducing additional features and improvements.

### Planned Features
- Improve reward shaping for the model.
- Implement a second model for comparison
- Improve state representation


## Developer Notes
- patience my child, naa sa mga 150+ pa before mag taas taas ang score
- Implement a new model first then make them integratable with the gui. THEN improve reward shaping


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