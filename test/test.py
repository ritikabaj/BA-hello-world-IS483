# ======================
# Setup + Safe Helpers
# ======================
import sys, re, warnings, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from packaging import version
import re

def range_to_float(x):
    """Convert range strings like '2-3' to average float"""
    if isinstance(x, str) and '-' in x:
        parts = x.split('-')
        try:
            a, b = float(parts[0]), float(parts[1])
            return (a + b) / 2
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def year_to_int(x):
    """Extract year number from strings like 'Year 3'"""
    if isinstance(x, str):
        m = re.search(r'\d+', x)
        if m:
            return int(m.group())
        return np.nan
    try:
        return int(x)
    except:
        return np.nan


warnings.filterwarnings("ignore")

print("="*70)
print("SMU Wellness — Visual EDA + Feature Engineering + Baseline Model")
print("This notebook keeps duplicates, adds visuals, clear printouts, and")
print("handles common errors (missing columns, package versions, NaNs).")
print("="*70)

# --------------- sklearn compatibility helpers ----------------
SKL = None
try:
    import sklearn
    SKL = sklearn.__version__
    print(f"[info] scikit-learn version: {SKL}")
except Exception as e:
    print("[warn] Could not detect scikit-learn version:", e)

def make_onehot_encoder():
    """Return an OneHotEncoder that works across sklearn versions."""
    try:
        # sklearn >= 1.2 supports sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older versions use sparse=
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_calibrator(estimator, method="isotonic", cv=3):
    """Return CalibratedClassifierCV that works for old/new sklearn APIs."""
    try:
        # new API (>=1.4) uses 'estimator='
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        # older API uses 'base_estimator='
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)

# --------------- plotting niceties ----------------
def annotate_bars(ax):
    """Add value labels above bars."""
    for p in ax.patches:
        try:
            val = p.get_height()
        except Exception:
            val = p.get_width()
        if pd.notna(val) and val != 0:
            ax.annotate(f"{int(val)}", (p.get_x()+p.get_width()/2, val),
                        ha='center', va='bottom', fontsize=9, rotation=0)

def explain(title, note):
    """Print a short explanation above a section."""
    print(f"\n--- {title} ---")
    print(note)


# ======================
# Load Data + Column Map
# ======================
import pandas as pd

# Load your Excel file
df = pd.read_excel('/Users/ritikabajpai/Desktop/SMU_Survey_Final.xlsx')

# Updated column maps based on actual column names in your file
KEY_COLUMNS = {
    'age': 'Age',
    'gender': 'Gender', 
    'year': 'YearOfStudy',
    'school': 'School',
    'intent_90d': 'Q27_27InTheNext90DaysHowLikelyAreYouToUseAnySmuWellnessServiceCounsellingWorkshopsEvents',
    'past_attend_count': 'Q28_28InThePast6MonthsHowManyWellnessEventsDidYouAttend',
    'no_show_count': 'Q29_29InThePast6MonthsHowManyWellnessEventsDidYouSignUpButNotAttendNoShow',
    'next_program': 'Q30_30IfYouCouldJoinExactlyOneWellnessProgramNextWhichTypeOfEventWouldYouPick',
    'preferred_format': 'Q31_31WhichFormatDoYouPreferMost',
    'preferred_time_windows': 'Q32_32WhichTimesWorkBestForYouSelectUpTo2',
    'barrier_time': 'Q33_lackOfTime',
    'barrier_cost': 'Q33_cost',
    'barrier_motivation': 'Q33_lackOfMotivationWillpower',
    'barrier_access': 'Q33_convenience',
}

RELEVANCE_COLUMNS = {
    'relevance_MHW': 'Q34_mentalHealthWeek',
    'relevance_Resilience': 'Q34_resilienceFramework',
    'relevance_ExamAngels': 'Q34_examAngels',
    'relevance_SCS': 'Q34_studentCareServices',
    'relevance_CosyHaven': 'Q34_cosyHaven',
    'relevance_Voices': 'Q34_voicesRoadshows',
    'relevance_PeerHelpers': 'Q34_peerHelpersRoadshows',
    'relevance_CareerCompass': 'Q34_careerCompass',
    'relevance_CARES': 'Q34_caresCorner'
}

use_cols = {**KEY_COLUMNS, **RELEVANCE_COLUMNS}

# Validate availability and rename to clean names
missing = [v for v in use_cols.values() if v not in df.columns]
if missing:
    raise KeyError(
        "The following expected survey columns are missing:\n"
        + "\n".join(f"- {m}" for m in missing) +
        "\nCheck the questionnaire wording/typos or update the map."
    )

data = df[list(use_cols.values())].rename(columns={v:k for k,v in use_cols.items()})
print(f"[ok] Selected {len(use_cols)} features from {len(data)} rows (duplicates retained).")


# Multi-select time windows
def multi_hot_counts(series, choices):
    s = (series.fillna('')
               .astype(str)
               .str.replace(r'\s*[/;]\s*', ', ', regex=True))
    exploded = s.str.split(',').explode().str.strip()
    exploded = exploded[exploded.ne('')]
    return exploded.value_counts().reindex(choices, fill_value=0)

time_choices = ['Weekday daytime','Weekday evening','Weekend daytime','Weekend evening']
plt.figure(figsize=(7,4))
ax = multi_hot_counts(data['preferred_time_windows'], time_choices).plot(kind='bar')
plt.title('Preferred Time Windows (multi-select)'); plt.ylabel('Mentions'); annotate_bars(ax); plt.tight_layout(); plt.show()

# Barriers
explain("Barriers (0–10)", 
        "Higher means a larger barrier. This highlights what mainly stops students "
        "(e.g., time vs motivation).")

barrier_cols = ['barrier_time','barrier_cost','barrier_motivation','barrier_access']
means = data[barrier_cols].apply(pd.to_numeric, errors='coerce').mean().sort_values()
plt.figure(figsize=(7,4))
ax = means.plot(kind='barh')
plt.title('Barriers — Mean (0–10)'); plt.xlabel('Mean score'); plt.tight_layout(); plt.show()

# Relevance of initiatives
explain("Initiatives relevance (1–10)", 
        "Which schemes feel most relevant right now? Use this to align programming "
        "and communications with perceived needs.")

rel_cols = [c for c in data.columns if c.startswith('relevance_')]
rel_means = data[rel_cols].apply(pd.to_numeric, errors='coerce').mean().sort_values(ascending=False)
plt.figure(figsize=(9,4))
ax = rel_means.plot(kind='bar')
plt.title('Initiatives Relevance — Mean (1–10)'); plt.ylabel('Mean score'); annotate_bars(ax); plt.tight_layout(); plt.show()

# Correlations
explain("Correlation (numeric only)", 
        "Blue=negative, yellow=positive. Use this to spot simple linear relationships "
        "(e.g., intent vs barriers). Not causal, just association.")
num_cols = ['age','intent_90d'] + barrier_cols + rel_cols
corr = data[num_cols].apply(pd.to_numeric, errors='coerce').corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='nearest')
plt.colorbar(); plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.index)
plt.title('Correlation heatmap'); plt.tight_layout(); plt.show()


# ==============================================
# Feature Engineering + Calibrated Logistic Reg
# (Verbose, explain-as-you-go version)
# ==============================================
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

def _make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _make_calibrator(estimator, method="isotonic", cv=3):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)

def _say(title, text):
    print(f"\n=== {title} ===")
    print(text)

# 1) Build engineered table exactly per spec
_say("Feature engineering",
    "- ordinals: past_attend_count→0/1/2/3; no_show_count→0/1/2\n"
    "- scaled numerics: intent/10, barriers/10, max relevance/10\n"
    "- multi-hot: preferred_time_windows (4 binaries)\n"
    "- one-hot: next_program, preferred_format, school, gender\n"
    "- derived indices: barrier_index, scheme_relevance_max")

def map_past(x):
    s = str(x).strip()
    if s in ['0','Zero','zero']: return 0
    if s in ['1','One','one']: return 1
    if '2-3' in s or '2 – 3' in s or '2 to 3' in s: return 2
    if '4+' in s or '4 +' in s or '4 or more' in s: return 3
    try:
        n = int(float(s))
        if n<=0: return 0
        if n==1: return 1
        if 2<=n<=3: return 2
        return 3
    except: return np.nan

def map_noshow(x):
    s = str(x).strip()
    if s in ['0','Zero','zero']: return 0
    if s in ['1','One','one']: return 1
    if '2+' in s or '2 +' in s or '2 or more' in s: return 2
    try:
        n = int(float(s))
        if n<=0: return 0
        if n==1: return 1
        return 2
    except: return np.nan

X = data.copy()
barrier_cols = ['barrier_time','barrier_cost','barrier_motivation','barrier_access']
rel_cols = [c for c in X.columns if c.startswith('relevance_')]

# Ordinals
X['past_attend_ord'] = X['past_attend_count'].map(map_past)
X['no_show_ord']     = X['no_show_count'].map(map_noshow)

# Scaled numerics & indices
X['intent_scaled']        = pd.to_numeric(X['intent_90d'], errors='coerce')/10.0
X['barrier_index']        = X[barrier_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)/10.0
X['scheme_relevance_max'] = X[rel_cols].apply(pd.to_numeric, errors='coerce').max(axis=1)/10.0

# Multi-hot: preferred time windows
time_choices = ['Weekday daytime','Weekday evening','Weekend daytime','Weekend evening']
def _split_multi(val):
    if pd.isna(val): return []
    s = re.sub(r'\s*[/;]\s*', ', ', str(val))
    return [t.strip() for t in s.split(',') if t.strip()]

for t in time_choices:
    X[f'time_{t.replace(" ","_").lower()}'] = X['preferred_time_windows'].apply(lambda v: 1 if t in _split_multi(v) else 0)

# Define feature roles
num_feats = ['intent_scaled','barrier_index','scheme_relevance_max','age','past_attend_ord','no_show_ord',
             'barrier_time','barrier_cost','barrier_motivation','barrier_access'] + rel_cols
cat_onehot = ['next_program','preferred_format','school','gender']
bin_feats = [c for c in X.columns if c.startswith('time_')]

# Show a compact data dictionary
_say("Data dictionary (compact)",
    f"Numeric (scaled/ordinal + raw): {len(num_feats)}\n"
    f"One-hot categories: {cat_onehot}\n"
    f"Binary time-window flags: {bin_feats}\n"
    "Tip: Missing numeric values will be median-imputed; categorical = mode.")

# 2) Proxy label & class balance
y = (pd.to_numeric(X['intent_90d'], errors='coerce') >= 7).astype(int)
pos, neg = int(y.sum()), int((1-y).sum())
_say("Proxy label",
    f"We temporarily label y=1 if intent_90d ≥ 7.\n"
    f"Class balance → positives={pos} ({pos/len(y):.1%}), negatives={neg} ({neg/len(y):.1%}).\n"
    "We set class_weight='balanced' so both classes are considered fairly.")

# 3) Splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"[split] train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}")

# 4) Preprocess
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('onehot',  _make_onehot_encoder())])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_onehot),
    ('bin', 'passthrough', bin_feats)
], remainder='drop')

# 5) Model + calibration (with explanation)
base = LogisticRegression(max_iter=2000, class_weight='balanced')
clf  = _make_calibrator(base, method='isotonic', cv=3)

_say("Calibration",
    "We wrap Logistic Regression with isotonic calibration so predicted "
    "probabilities behave like real-world chances (e.g., 0.70 ≈ 70% attend).")

pipe = Pipeline([('prep', preprocessor), ('clf', clf)])

try:
    pipe.fit(X_train, y_train)
    print("[ok] Trained + calibrated.")
except Exception as e:
    _say("Training error",
        "Common causes:\n"
        "- Non-numeric values in numeric columns that couldn’t be coerced\n"
        "- Columns that became all-NaN after selection\n"
        "- Package version mismatch\n"
        f"Raw error: {e}")
    raise

# 6) Metrics + explanations
def _report(split, Xs, ys):
    p = pipe.predict_proba(Xs)[:,1]
    yhat = (p >= 0.5).astype(int)
    print(f"\n--- {split} metrics ---")
    print(f"AUROC : {roc_auc_score(ys, p):.3f}  → ranking quality; 0.5=chance, 1.0=perfect")
    print(f"PR-AUC: {average_precision_score(ys, p):.3f}  → precision–recall area; useful with imbalance")
    print(f"F1@0.5: {f1_score(ys, yhat):.3f}  → balance of precision & recall at 0.5 threshold")
    print(f"Brier : {brier_score_loss(ys, p):.3f}  → prob. calibration error; lower is better (≈0 best)")
    return p

p_tr = _report("Train", X_train, y_train)
p_va = _report("Valid", X_valid, y_valid)
p_te = _report("Test ", X_test,  y_test)

# 7) Recommend operational buckets on the validation set
_say("Operating policy (validation)",
    "We form three action buckets on validation predictions:\n"
    "  • LOW   : p < 0.35  → deprioritize or light-touch nudge\n"
    "  • MEDIUM: 0.35–0.70 → normal comms; consider incentives\n"
    "  • HIGH  : p ≥ 0.70  → priority invites, personalized nudges")

def bucketize(p):
    cats = pd.cut(p, bins=[-1,0.35,0.70,1.01], labels=['LOW','MED','HIGH'])
    return cats

val_buckets = bucketize(p_va)
val_summary = pd.crosstab(val_buckets, y_valid, rownames=['bucket'], colnames=['actual'])
print("\n[validation bucket vs actual]")
print(val_summary)
print("\nInterpretation:")
print("- Within HIGH, a greater share of actual 1s means your threshold is useful for targeting.")
print("- Adjust 0.35/0.70 to trade off coverage vs precision for campaigns.")

# 8) Reliability (calibration) curve – with plain-English caption
try:
    prob_true, prob_pred = [], []
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_valid, p_va, n_bins=10, strategy='quantile')
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1],[0,1],'--', label='Perfect')
    plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency'); plt.title('Calibration (Validation)')
    plt.legend(); plt.tight_layout(); plt.show()
    print("If the blue dots hug the dashed line, your probabilities are well-calibrated.")
except Exception:
    pass

# 9) “Top drivers” (global) with a short, readable mapping
try:
    clf_cv = pipe.named_steps['clf']
    coefs = np.mean([est.coef_.ravel() for est in clf_cv.calibrated_classifiers_], axis=0)
    # Get expanded feature names
    try:
        ohe = pipe.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(cat_onehot))
    except Exception:
        ohe = pipe.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names(cat_onehot))
    final_features = list(num_feats) + ohe_names + bin_feats

    drivers = pd.Series(coefs, index=final_features).sort_values(key=np.abs, ascending=False)
    top = drivers.head(15).rename("coef")
    # Friendly interpretation column
    def _how(f, c):
        direction = "↑ raises" if c>0 else "↓ lowers"
        if f.startswith('time_'):
            return f"Time-window flag ({f.replace('time_','').replace('_',' ')}) {direction} p(attend)"
        if f in ['intent_scaled','barrier_index','scheme_relevance_max','age','past_attend_ord','no_show_ord']:
            return f"{f} {direction} p(attend)"
        if 'relevance_' in f or 'barrier_' in f:
            return f"{f} {direction} p(attend)"
        return f"{f} {direction} p(attend)"
    interp = pd.DataFrame({
        "feature": top.index,
        "coef": top.values,
        "explanation": [ _how(f,c) for f,c in top.items() ]
    })
    print("\nTop drivers (global, logistic-regression coefficients):")
    display(interp)
    print("Note: coefficients act on the log-odds. Bigger |coef| = stronger influence.")
except Exception as e:
    _say("Driver summary skipped", f"Could not compute coefficients: {e}")

    # ============================================
# FIRST: CREATE TRAIN/VALID/TEST SPLITS  
# ============================================

from sklearn.model_selection import train_test_split
import pandas as pd

print("\n" + "="*60)
print("STEP 1: Creating Train/Valid/Test Splits")
print("="*60)

# Create target variable (likelihood to attend based on intent)
data['target'] = (data['intent_90d'] >= 7).astype(int)

print(f"\nTotal samples: {len(data)}")
print(f"Target distribution:")
print(f"  Likely to attend (y=1): {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
print(f"  Not likely (y=0): {(1-data['target']).sum()} ({(1-data['target']).mean()*100:.1f}%)")

# Stratified split: 70% train, 15% validation, 15% test
# This maintains school distribution across all splits
train_data, temp_data = train_test_split(
    data, 
    test_size=0.30,  # 30% for valid+test
    stratify=data['school'],  # <-- DEBIASING PART 1: Maintains proportions
    random_state=42
)

# Split temp into validation and test (50/50 = 15% each of total)
valid_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    stratify=temp_data['school'],
    random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
print(f"  Valid: {len(valid_data)} ({len(valid_data)/len(data)*100:.1f}%)")
print(f"  Test:  {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")

# Verify stratification worked
print("\nSchool distribution maintained across splits:")
for split_name, split_df in [('Train', train_data), ('Valid', valid_data), ('Test', test_data)]:
    dist = split_df['school'].value_counts(normalize=True) * 100
    print(f"\n{split_name}:")
    for school, pct in dist.items():
        print(f"  {school}: {pct:.1f}%")

print("\n✅ Data split complete! Now ready for debiasing...")


# ============================================
# PART B: DEBIASING WITH EQUAL DISTRIBUTION TARGET
# ============================================

print("\n" + "="*60)
print("APPLYING DEBIASING - Target: EQUAL Distribution")
print("="*60)

import pandas as pd
import numpy as np

def calculate_debiasing_weights_equal(df):
    """
    Calculate post-stratification weights to achieve EQUAL distribution.
    
    Target: All schools should have equal representation (1/n_schools each)
    Formula: weight = (1/n_schools) / sample_share
    
    This ensures the model learns EQUALLY from ALL schools!
    """
    # Get sample distribution (what we actually have)
    sample_dist = df['school'].value_counts(normalize=True)
    
    # Target distribution: EQUAL for all schools
    n_schools = len(sample_dist)
    target_share = 1.0 / n_schools  # Each school should be equal
    
    # Calculate weight for each student
    weights = pd.Series(index=df.index, dtype=float)
    
    print("\nWeight calculations (Target: Equal distribution):")
    print("-" * 60)
    print(f"Target per school: {target_share*100:.1f}% (equal representation)\n")
    
    for school in df['school'].unique():
        if pd.notna(school):
            sample_share = sample_dist.get(school, 0)
            
            if sample_share > 0:
                weight = target_share / sample_share
                weights[df['school'] == school] = weight
                
                effect = "⬇️ DOWNWEIGHT" if weight < 1 else "⬆️ UPWEIGHT"
                change = ((weight - 1.0) * 100)
                print(f"{school:10s}:")
                print(f"  Sample:      {sample_share*100:5.1f}%")
                print(f"  Target:      {target_share*100:5.1f}%")
                print(f"  → Weight:    {weight:.3f} {effect} ({change:+.1f}%)")
            else:
                weights[df['school'] == school] = 1.0
        else:
            weights[df['school'].isna()] = 1.0
    
    return weights

# Apply debiasing weights with EQUAL distribution target
print("\n" + "="*60)
print("Applying weights to train/valid/test splits...")
print("="*60)

train_data['sample_weight'] = calculate_debiasing_weights_equal(train_data)
valid_data['sample_weight'] = calculate_debiasing_weights_equal(valid_data)
test_data['sample_weight'] = calculate_debiasing_weights_equal(test_data)

print("\n✅ Debiasing weights successfully applied!")
print("\nWeight summary (training set):")
print("=" * 60)
weight_summary = train_data.groupby('school')['sample_weight'].first().sort_values()
for school, weight in weight_summary.items():
    bar = "█" * int(weight * 10)  # Adjusted scale
    effect = "⬇️" if weight < 1 else "⬆️"
    print(f"  {school:10s}: {weight:.3f} {effect} {bar}")

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("  Schools with > 14.3% sample → DOWNWEIGHTED")
print("  Schools with < 14.3% sample → UPWEIGHTED")
print("  Result: Model learns EQUALLY from all schools!")
print("=" * 60)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

# 1. Data Cleaning: Convert ranges/strings to numbers for numeric columns
def range_to_float(x):
    if isinstance(x, str) and '-' in x:
        a, b = x.split('-')
        return (float(a) + float(b)) / 2
    try:
        return float(x)
    except:
        return np.nan

def year_to_int(x):
    if isinstance(x, str):
        import re
        m = re.search(r'\d+', x)
        if m:
            return int(m.group())
        return np.nan
    return x

for df in [train_data, valid_data, test_data]:
    df['past_attend_count'] = df['past_attend_count'].apply(range_to_float)
    df['no_show_count'] = df['no_show_count'].apply(range_to_float)
    df['year'] = df['year'].apply(year_to_int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')  # In case age has similar issues

# 2. Features/Target (copy from your columns exactly)
FEATURES = [
     'past_attend_count', 'no_show_count',
    'next_program', 'preferred_format', 'preferred_time_windows',
    'barrier_time', 'barrier_cost', 'barrier_motivation', 'barrier_access',
    'relevance_MHW', 'relevance_Resilience', 'relevance_ExamAngels', 'relevance_SCS',
    'relevance_CosyHaven', 'relevance_Voices', 'relevance_PeerHelpers',
    'relevance_CareerCompass', 'relevance_CARES',
    'school', 'age', 'gender', 'year'
]
TARGET = 'target'

numeric_features = [
     'past_attend_count', 'no_show_count', 'age', 'year',
    'relevance_MHW', 'relevance_Resilience', 'relevance_ExamAngels', 'relevance_SCS',
    'relevance_CosyHaven', 'relevance_Voices', 'relevance_PeerHelpers',
    'relevance_CareerCompass', 'relevance_CARES'
]
categorical_features = [
    'next_program', 'preferred_format', 'preferred_time_windows',
    'barrier_time', 'barrier_cost', 'barrier_motivation', 'barrier_access',
    'school', 'gender'
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 3. Prepare data
X_train = train_data[FEATURES]
y_train = train_data[TARGET]
w_train = train_data["sample_weight"]
X_valid = valid_data[FEATURES]
y_valid = valid_data[TARGET]
w_valid = valid_data["sample_weight"]

# 4. Build and fit pipeline
baseline_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42))
])
baseline_pipe.fit(X_train, y_train, clf__sample_weight=w_train)

# 5. Predict and evaluate
y_pred_valid = baseline_pipe.predict(X_valid)
y_prob_valid = baseline_pipe.predict_proba(X_valid)[:, 1]
print("ROC-AUC:", roc_auc_score(y_valid, y_prob_valid, sample_weight=w_valid))
print("F1 Score:", f1_score(y_valid, y_pred_valid, sample_weight=w_valid))
import numpy as np
import pandas as pd

# --- CLEANING: make sure these match your earlier usage ---
def range_to_float(x):
    if isinstance(x, str) and '-' in x:
        a, b = x.split('-')
        return (float(a) + float(b)) / 2
    try:
        return float(x)
    except:
        return np.nan

def year_to_int(x):
    if isinstance(x, str):
        import re
        m = re.search(r'\d+', x)
        if m:
            return int(m.group())
        return np.nan
    return x

# Apply to your survey DataFrame (assume it's called 'data')
for col in ['past_attend_count', 'no_show_count']:
    data[col] = data[col].apply(range_to_float)
data['year'] = data['year'].apply(year_to_int)
data['age'] = pd.to_numeric(data['age'], errors='coerce')

# --- PREDICTION: grab random rows and predict ---
SAMPLE = data.sample(n=5, random_state=42)         # Take 5 random students

X_sample = SAMPLE[FEATURES]                        # Extract feature columns
probs = baseline_pipe.predict_proba(X_sample)[:,1] # Predicted probability to attend
attend_pred = baseline_pipe.predict(X_sample)      # 0/1 predicted attendance

SAMPLE = SAMPLE.copy()                             # Attach result columns
SAMPLE['predicted_attendance_prob'] = probs        # Probability output (0.0-1.0)
SAMPLE['predicted_attend'] = attend_pred           # 1 = attend, 0 = not attend

print(SAMPLE[['predicted_attendance_prob', 'predicted_attend'] + FEATURES])


# ============================================================================
# COMPLETE SOLUTION: XGBoost Model + Feature Importance + Predictions
# ============================================================================

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score

print("="*70)
print("STEP 5: Enhanced Model - XGBoost Training")
print("="*70)

# 1. Preprocess data
# CLEAN DATA FIRST - Apply the same cleaning to X_train, X_valid, X_test
print("\nCleaning data before preprocessing...")

# Helper functions are already defined at the top of your script
# Apply cleaning to X_train
X_train = X_train.copy()
X_train['past_attend_count'] = X_train['past_attend_count'].apply(range_to_float)
X_train['no_show_count'] = X_train['no_show_count'].apply(range_to_float)
X_train['year'] = X_train['year'].apply(year_to_int)
X_train['age'] = pd.to_numeric(X_train['age'], errors='coerce')

# Apply cleaning to X_valid
X_valid = X_valid.copy()
X_valid['past_attend_count'] = X_valid['past_attend_count'].apply(range_to_float)
X_valid['no_show_count'] = X_valid['no_show_count'].apply(range_to_float)
X_valid['year'] = X_valid['year'].apply(year_to_int)
X_valid['age'] = pd.to_numeric(X_valid['age'], errors='coerce')

# Apply cleaning to X_test
X_test = X_test.copy()
X_test['past_attend_count'] = X_test['past_attend_count'].apply(range_to_float)
X_test['no_show_count'] = X_test['no_show_count'].apply(range_to_float)
X_test['year'] = X_test['year'].apply(year_to_int)
X_test['age'] = pd.to_numeric(X_test['age'], errors='coerce')

print("✓ Data cleaned successfully!")

# NOW proceed with preprocessing
X_train_prep = preprocessor.fit_transform(X_train)
X_valid_prep = preprocessor.transform(X_valid)
X_test_prep = preprocessor.transform(X_test)


# 2. Handle class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

print(f"\nClass Balance: {neg_count} negative, {pos_count} positive samples")
print(f"Scale weight: {scale_pos_weight:.2f}")

# 3. Simplified XGBoost model (faster training)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1, 2]
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False
)

print("\nTraining XGBoost with GridSearch (this may take 2-5 minutes)...")
grid_search = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=0)
grid_search.fit(X_train_prep, y_train, sample_weight=w_train)
best_xgb = grid_search.best_estimator_

# 4. Evaluate
y_prob_val = best_xgb.predict_proba(X_valid_prep)[:, 1]
y_prob_test = best_xgb.predict_proba(X_test_prep)[:, 1]

# 5. TOP 5 FEATURES
print("\n" + "="*70)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*70)

try:
    cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    all_features = list(numeric_features) + list(cat_names)
except:
    all_features = [f'feature_{i}' for i in range(X_train_prep.shape[1])]

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False).head(5)

for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i}. {row['Feature']:<40s} {row['Importance']:.4f}")

# 6. PREDICT RANDOM STUDENTS
print("\n" + "="*70)
print("PREDICTION: 5 RANDOM STUDENTS FROM SURVEY")
print("="*70)

# Sample 5 random students
sample_students = data.sample(n=5, random_state=42)
X_sample = sample_students[FEATURES]
X_sample_prep = preprocessor.transform(X_sample)

# Get predictions
probs = best_xgb.predict_proba(X_sample_prep)[:, 1]
preds = best_xgb.predict(X_sample_prep)

# Create results DataFrame
results = pd.DataFrame({
    'Student_ID': sample_students.index,
    'School': sample_students['school'],
    'Attend_Probability': [f"{p*100:.1f}%" for p in probs],
    'Prediction': ['WILL ATTEND' if p == 1 else 'WON\'T ATTEND' for p in preds]
})

print(results.to_string(index=False))

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

# Save model for later use (optional)
print("\nSaving best model...")
import pickle
with open('attendance_model.pkl', 'wb') as f:
    pickle.dump({'model': best_xgb, 'preprocessor': preprocessor}, f)
print("Model saved as 'attendance_model.pkl'")

# ============================================================================
# COMPLETE SOLUTION: XGBoost with Threshold Optimization & All Metrics (DEBUGGED)
# ============================================================================

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                             precision_score, recall_score, confusion_matrix)
import sys

print("="*80)
print("STEP 5: Enhanced Model - XGBoost Training with Hyperparameter Tuning")
print("="*80)

print("\n[SECTION 1] Data Preprocessing")
print("-" * 80)

X_train_prep = preprocessor.fit_transform(X_train)
X_valid_prep = preprocessor.transform(X_valid)
X_test_prep = preprocessor.transform(X_test)

print(f"✓ Training features shape: {X_train_prep.shape}")
print(f"✓ Validation features shape: {X_valid_prep.shape}")
print(f"✓ Test features shape: {X_test_prep.shape}")

print("\n[SECTION 2] Handling Class Imbalance")
print("-" * 80)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

print(f"Class distribution in training data:")
print(f"  • Won't Attend (0): {neg_count} students ({neg_count/(neg_count+pos_count)*100:.1f}%)")
print(f"  • Will Attend (1):  {pos_count} students ({pos_count/(neg_count+pos_count)*100:.1f}%)")
print(f"  • Scale Weight: {scale_pos_weight:.2f}")

print("\n[SECTION 3] Defining Hyperparameter Search Space")
print("-" * 80)

param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1, 2]
}

total_combinations = 2 * 2 * 2 * 2 * 2 * 2 * 2
print(f"Testing {total_combinations} combinations with 3-fold CV")

print("\n[SECTION 4] Training XGBoost with GridSearchCV")
print("-" * 80)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=0
)

print("Training in progress...")
grid_search.fit(X_train_prep, y_train, sample_weight=w_train)
best_xgb = grid_search.best_estimator_

print(f"✓ Training complete!")
print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")

print("\n[SECTION 5] Best Hyperparameters Found")
print("-" * 80)
for param, value in grid_search.best_params_.items():
    print(f"  • {param:20s}: {value}")

print("\n[SECTION 6] Generating Predictions")
print("-" * 80)

y_prob_val = best_xgb.predict_proba(X_valid_prep)[:, 1]
y_prob_test = best_xgb.predict_proba(X_test_prep)[:, 1]

print("✓ Predictions generated for validation and test sets")
sys.stdout.flush()

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)
print("Finding optimal prediction threshold to maximize F1 score...\n")

thresholds = np.arange(0.0, 1.01, 0.05)
f1_scores_val = []
recall_scores_val = []
precision_scores_val = []

for threshold in thresholds:
    y_pred_val_thresh = (y_prob_val >= threshold).astype(int)
    f1_val = f1_score(y_valid, y_pred_val_thresh, zero_division=0, sample_weight=w_valid)
    f1_scores_val.append(f1_val)
    
    prec = precision_score(y_valid, y_pred_val_thresh, zero_division=0, sample_weight=w_valid)
    rec = recall_score(y_valid, y_pred_val_thresh, zero_division=0, sample_weight=w_valid)
    precision_scores_val.append(prec)
    recall_scores_val.append(rec)

# Find optimal threshold
best_idx_val = np.argmax(f1_scores_val)
best_threshold_f1 = thresholds[best_idx_val]
best_f1_val = f1_scores_val[best_idx_val]

print(f"Optimal threshold (maximizes F1): {best_threshold_f1:.2f}")
print(f"  Validation F1 at this threshold: {best_f1_val:.4f}")

# Apply optimal threshold to test set
y_pred_test_optimal = (y_prob_test >= best_threshold_f1).astype(int)




# ============================================================================
# SECTION 8: CONFUSION MATRIX - DEBUGGED
# ============================================================================

print("\n" + "="*80)
print("CONFUSION MATRIX (Test Set)")
print("="*80)

print("DEBUG: Calculating confusion matrix...")
try:
    cm = confusion_matrix(y_test, y_pred_test_optimal)
    print(f"DEBUG: Confusion matrix shape: {cm.shape}")
    print(f"""
              Predicted: No    Predicted: Yes
Actual: No        {cm[0,0]}            {cm[0,1]}
Actual: Yes       {cm[1,0]}            {cm[1,1]}

True Negatives (TN):  {cm[0,0]}  - Correctly predicted won't attend
False Positives (FP): {cm[0,1]}  - Incorrectly predicted will attend
False Negatives (FN): {cm[1,0]}  - Incorrectly predicted won't attend
True Positives (TP):  {cm[1,1]}  - Correctly predicted will attend
    """)
except Exception as e:
    print(f"❌ ERROR in confusion matrix: {e}")
    import traceback
    traceback.print_exc()

sys.stdout.flush()



# ============================================================================
# SECTION 9: TOP 5 FEATURES - DEBUGGED
# ============================================================================

print("\n" + "="*80)
print("TOP 5 MOST IMPORTANT FEATURES")
print("="*80)
print("These features have the strongest influence on attendance predictions:\n")

print("DEBUG: Extracting feature names...")
try:
    # Try to get feature names
    try:
        cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(cat_names)
        print(f"DEBUG: Got {len(all_features)} feature names")
    except Exception as e:
        print(f"DEBUG: Could not get feature names from preprocessor: {e}")
        all_features = [f'feature_{i}' for i in range(X_train_prep.shape[1])]
        print(f"DEBUG: Using generic names for {len(all_features)} features")

    # Get feature importances
    importances = best_xgb.feature_importances_
    print(f"DEBUG: Got {len(importances)} importance values")
    
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(5)

    for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
        print(f"  {i}. {row['Feature']:<40s} {row['Importance']:.4f}")
        
except Exception as e:
    print(f"❌ ERROR displaying features: {e}")
    import traceback
    traceback.print_exc()

sys.stdout.flush()

# ============================================================================
# SECTION 10: RANDOM 5 STUDENTS PREDICTIONS - DEBUGGED
# ============================================================================

print("\n" + "="*80)
print("PREDICTION: 5 RANDOM STUDENTS FROM SURVEY")
print("="*80)
print("Real-world predictions with attendance probability:\n")

print("DEBUG: Sampling students...")
try:
    # Make sure 'data' variable exists
    print(f"DEBUG: data shape: {data.shape}")
    print(f"DEBUG: FEATURES: {FEATURES}")
    
    sample_students = data.sample(n=5, random_state=42)
    print(f"DEBUG: Sampled {len(sample_students)} students")
    
    X_sample = sample_students[FEATURES]
    print(f"DEBUG: X_sample shape: {X_sample.shape}")
    
    X_sample_prep = preprocessor.transform(X_sample)
    print(f"DEBUG: X_sample_prep shape: {X_sample_prep.shape}")

    probs = best_xgb.predict_proba(X_sample_prep)[:, 1]
    print(f"DEBUG: Generated {len(probs)} predictions")
    
    preds_sample = (probs >= best_threshold_f1).astype(int)

    results = pd.DataFrame({
        'Student_ID': sample_students.index,
        'School': sample_students['school'] if 'school' in sample_students.columns else ['N/A']*5,
        'Attend_Probability': [f"{p*100:.1f}%" for p in probs],
        'Prediction': ['WILL ATTEND ✓' if p == 1 else 'WON\'T ATTEND ✗' for p in preds_sample]
    })

    print(results.to_string(index=False))
    
except Exception as e:
    print(f"❌ ERROR making predictions: {e}")
    import traceback
    traceback.print_exc()

sys.stdout.flush()



# ============================================================
# SAVING MODEL FOR DEPLOYMENT
# ============================================================
print("\n" + "="*80)
print("SAVING MODEL FOR DEPLOYMENT")
print("="*80)
print("DEBUG: Attempting to save model...")

try:
    import pickle
    
    model_package = {
        'model': best_xgb,
        'preprocessor': preprocessor,
        'features': FEATURES,
        'optimal_threshold': best_threshold_f1,
        # Remove or comment out this line if metrics aren't needed:
        # 'metrics': metrics_test_optimal
    }
    
    with open('attendance_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("✅ Model saved as 'attendance_model.pkl'")
    print(f"   Optimal threshold saved: {best_threshold_f1:.2f}")
    
except Exception as e:
    print(f"❌ ERROR saving model: {e}")


