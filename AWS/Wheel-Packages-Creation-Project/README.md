# Wheel Packages Creation

Purpose: For AWS ETL
Objective: To create wheel packages for AWS ETL projects.

## How to use

1. Create local Python virtual environment:

    ```bash
    python3 -m venv .venv
    ```

2. Activate the virtual environment:

    ```bash
    .venv\Scripts\Activate.ps1
    ```

3. Install pip and setuptools:

    ```bash
    pip install --upgrade pip setuptools wheel
    ```

4. Create a new directory for packaging:

    ```bash
    mkdir glue_libs
    cd glue_libs
    ```

5. Install the packages locally into this directory

    ```bash
    pip install pandas openpyxl -t .
    ```

6. Build the wheel package

    ```bash
    pip wheel pandas openpyxl -w ./wheelhouse
    ```

7. Upload the entire 'wheelhouse' content, only the .whl files, to your AWS S3 bucket.
   E.g. `s3://<your-bucket-name>/wheelhouse/`
8. In your AWS Glue job, specify the S3 path to the wheel files in the "Python library path" field.
