# PIK3CA: Mutation in breast cancer

# Installation

## Mambaforge
Download the `mamba` installation file corresponding to your system from here:
https://github.com/conda-forge/miniforge#mambaforge

When the fine is downloaded, go to the directory that the file is downloaded and run:

```bash
bash <FILE_NAME>
```
For example:
```bash
bash Mambaforge-MacOSX-x86_64.sh
```

## Environment installation
After the mamba installation, open a new terminal. Go to this package folder and run:
```bash
make install
```

After the installation, activate the environment:

```bash
mamba activate chowder
```

## Environment update
If you change any of the environment files, you can update your environment by running:
```bash
make update
```

# Test

```