# Running within this codebase

### Prerequisites

-   Python 3.7 or higher installed on your system
-   VS Code with the Python extension installed

### Step 1: Create a Virtual Environment

Open a terminal/command prompt in this directory.

Create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

**On Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**On macOS/Linux:**

```bash
source .venv/bin/activate
```

### Step 3: Install Required Packages

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Run the script

```bash
python main.py
```

### Important Notes

-   **Data File Path:** The notebook references a file path `/Users/brendankhow/Documents/GitHub/BA-hello-world-IS483/SMU_Survey_Final.xlsx`. You'll need to update this path to point to the correct location of your data file.

-   **Deactivating:** When you're done working, you can deactivate the virtual environment by running:

    ```bash
    deactivate
    ```

-   **Reactivating:** Next time you work on this project, just activate the virtual environment again (Step 2) and open VS Code.

### Required Packages

The `requirements.txt` file includes:

-   `pandas` - Data manipulation and analysis
-   `numpy` - Numerical computing
-   `matplotlib` - Plotting library
-   `seaborn` - Statistical data visualization
-   `openpyxl` - Excel file reading/writing support for pandas
