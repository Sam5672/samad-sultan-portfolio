"""
analyze.py — Central-bank statement analyzer (US / Australia / UK).

Trains three DistilBERT regression models at startup using HF Trainer,
then exposes analyze_statement() which scores every sentence in a
central-bank statement and predicts the 60-day yield direction from
sentiment alone.

Inputs:
    country   (str) : "US", "United States", "Pakistan",
                      "Australia", "UK", "United Kingdom"
    statement (str) : raw central-bank statement text

Outputs (dict):
    avg_score        (float)
    hawk_count       (int)
    dove_count       (int)
    neutral_count    (int)
    total_sentences  (int)
    scored_sentences (list[dict]) : [{"text": str, "score": float}, ...]
    prediction       (dict)       : {"change": float, "lower": float,
                                     "upper": float, "direction": str}
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(24)
np.random.seed(24)
torch.manual_seed(24)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
device      = torch.device("cpu")
_US_EPOCHS  = 30
_AU_EPOCHS  = 10
_UK_EPOCHS  = 10
_BATCH      = 8
_LR         = 2e-5
_MAX_LEN    = 128

# ── Per-country regression coefficients (sentiment-only) ─────────────────────
_REG = {
    "US": dict(intercept=-0.0295, coef_sentiment=2.05,
               mean_sentiment=0.50, residual_std=0.28),
    "AU": dict(intercept=-0.02,   coef_sentiment=1.80,
               mean_sentiment=0.50, residual_std=0.30),
    "UK": dict(intercept=-0.02,   coef_sentiment=1.80,
               mean_sentiment=0.50, residual_std=0.30),
}
_CI_MULTIPLIER = 1.2

# ── Skip patterns ─────────────────────────────────────────────────────────────
SKIP_US = [
    "Voting for", "Voting against", "Vice Chair", "Chair;",
    "media inquiries", "email", "Implementation Note",
    "For release", "press@", "call 202",
    "Brainard", "Bullard", "Bowman", "Harker", "Mester",
    "Kashkari", "George", "Kaplan", "Clarida", "Waller",
    "Jefferson", "Cook", "Hammack", "Logan", "Paulson",
    "Miran", "Barr", "Williams", "Powell",
]

SKIP_AU = [
    "Voting for", "Voting against", "media inquiries", "email",
    "Communications Department", "For release",
    "Reserve Bank of Australia SYDNEY",
    "Bullock", "Lowe", "Kent", "Ellis", "Debelle",
    "Hunter", "Kohler", "Hauser", "phone", "rbainfo",
]

SKIP_UK = [
    "Voting for", "Voting against", "media inquiries", "email",
    "Bank of England Act", "For release", "Treasury representative",
    "Bailey", "Broadbent", "Cunliffe", "Haskel", "Mann",
    "Pill", "Ramsden", "Dhingra", "Tenreyro", "Breeden",
    "Lombardelli", "Taylor", "Greene",
    "Sam Beckett", "David Roberts", "Jonathan Bewes",
]

# ── Training data — US (88 sentences) ────────────────────────────────────────
_TRAINING_US = [
    ("The Committee decided to raise the target range for the federal funds rate by 75 basis points", 0.95),
    ("The Committee decided to raise the target range for the federal funds rate by 50 basis points", 0.85),
    ("The Committee decided to raise the target range for the federal funds rate by 25 basis points", 0.75),
    ("The Committee decided to maintain the target range for the federal funds rate", 0.5),
    ("The Committee decided to lower the target range for the federal funds rate by 25 basis points", 0.25),
    ("The Committee decided to lower the target range for the federal funds rate by 50 basis points", 0.15),
    ("The Committee expects it will soon be appropriate to raise the target range for the federal funds rate", 0.85),
    ("It would be premature to begin lowering rates at this time", 0.8),
    ("The Committee anticipates that ongoing increases in the target range will be appropriate", 0.9),
    ("The Committee is strongly committed to returning inflation to its 2 percent objective", 0.8),
    ("The Committee is prepared to raise rates further if appropriate", 0.85),
    ("Additional policy firming may be appropriate", 0.8),
    ("The Committee will continue reducing its holdings of Treasury securities and agency debt", 0.75),
    ("The Committee decided to continue to reduce the monthly pace of its net asset purchases", 0.7),
    ("Ongoing increases in the target range will be appropriate to bring inflation down", 0.85),
    ("The Committee will take into account the cumulative tightening of monetary policy", 0.65),
    ("The Committee is prepared to adjust the stance of monetary policy as appropriate if risks emerge", 0.35),
    ("The Committee would be prepared to adjust the stance of monetary policy as appropriate", 0.4),
    ("The Committee judges that the risks have become more balanced", 0.3),
    ("The risks to achieving its employment and inflation goals are roughly in balance", 0.3),
    ("The Committee is attentive to the risks to both sides of its dual mandate", 0.35),
    ("In considering additional adjustments the Committee will carefully assess incoming data", 0.4),
    ("The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence", 0.55),
    ("Inflation remains elevated", 0.8),
    ("Inflation remains well above the Committee's 2 percent objective", 0.85),
    ("Inflation remains somewhat elevated", 0.65),
    ("Inflation continues to run well above the longer-run goal", 0.85),
    ("Core inflation remains elevated and persistent", 0.85),
    ("Price pressures remain broad based across sectors", 0.8),
    ("Inflation has shown little progress toward the 2 percent objective", 0.85),
    ("Supply and demand imbalances have continued to contribute to elevated levels of inflation", 0.75),
    ("Wage growth remains elevated relative to what would be consistent with price stability", 0.8),
    ("Housing costs continue to put upward pressure on overall inflation", 0.75),
    ("Inflation expectations risk becoming unanchored", 0.9),
    ("Inflation has eased over the past year but remains above the Committee's target", 0.55),
    ("Inflation has made progress toward the Committee's 2 percent objective", 0.3),
    ("Inflation has made further progress toward the Committee's 2 percent objective", 0.25),
    ("Inflation has declined significantly from its peak", 0.2),
    ("The disinflationary process has continued", 0.2),
    ("Inflation has come down considerably over the past year", 0.25),
    ("Longer-term inflation expectations remain well anchored", 0.35),
    ("Inflation has moved closer to the Committee's 2 percent objective", 0.3),
    ("The labor market remains extremely tight", 0.8),
    ("Job gains have been robust in recent months", 0.75),
    ("Job gains have been strong", 0.7),
    ("The unemployment rate has remained low", 0.65),
    ("Demand continues to outstrip supply in the labor market", 0.8),
    ("Consumer spending has been particularly strong", 0.75),
    ("Consumer spending has been surprisingly resilient", 0.7),
    ("Job gains have been solid in recent months and the unemployment rate has declined substantially", 0.7),
    ("Job gains have moderated", 0.35),
    ("Job gains have slowed", 0.3),
    ("The unemployment rate has moved up but remains low", 0.35),
    ("The labor market is showing signs of softening", 0.25),
    ("Job gains have remained low and the unemployment rate has shown some signs of stabilization", 0.35),
    ("Employment gains have slowed significantly", 0.2),
    ("Economic activity has been expanding at a strong pace", 0.75),
    ("Economic activity has been expanding at a solid pace", 0.65),
    ("Available indicators suggest that economic activity has been expanding at a solid pace", 0.65),
    ("The economy has been stronger than expected", 0.75),
    ("Recent indicators suggest that economic activity has continued to expand at a solid pace", 0.65),
    ("Indicators of economic activity and employment have continued to strengthen", 0.7),
    ("Economic activity has slowed from its strong pace", 0.35),
    ("Economic activity has slowed considerably", 0.2),
    ("Growth has moderated from its earlier pace", 0.3),
    ("Recent indicators point to modest growth in spending and production", 0.35),
    ("The path of the economy continues to depend on the course of the virus", 0.3),
    ("Economic activity expanded at a moderate rate", 0.4),
    ("Overall financial conditions remain accommodative", 0.3),
    ("Financial conditions have tightened significantly", 0.25),
    ("Credit conditions have become more restrictive for households and businesses", 0.25),
    ("Tighter credit conditions for households and businesses are likely to weigh on economic activity", 0.2),
    ("The Federal Reserve's ongoing purchases will continue to foster smooth market functioning and accommodative financial conditions", 0.3),
    ("The Committee will continue reducing its holdings of Treasury securities", 0.7),
    ("The Committee decided to begin reducing its holdings at a pace of 95 billion per month", 0.75),
    ("The Committee decided to slow the pace of decline of its securities holdings", 0.35),
    ("Beginning in February the Committee will increase its holdings of Treasury securities", 0.2),
    ("The Committee will increase its holdings of agency mortgage-backed securities", 0.2),
    ("Risks to the economic outlook remain including from new variants of the virus", 0.25),
    ("Uncertainty about the economic outlook has increased", 0.3),
    ("Uncertainty about the economic outlook remains elevated", 0.35),
    ("The Committee sees the risks to achieving its goals as roughly in balance", 0.4),
    ("Risks to the economic outlook are weighted to the downside", 0.15),
    ("Upside risks to inflation remain", 0.8),
    ("The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run", 0.5),
    ("The Committee will continue to monitor the implications of incoming information for the economic outlook", 0.5),
    ("The Committee's assessments will take into account a wide range of information", 0.5),
    ("In assessing the appropriate stance of monetary policy the Committee will continue to monitor", 0.5),
]

# ── Training data — AU ────────────────────────────────────────────────────────
_TRAINING_AU = [
    ("The Board decided to increase the cash rate target by 50 basis points", 0.92),
    ("The Board decided to increase the cash rate target by 25 basis points", 0.80),
    ("At its meeting today the Board decided to increase the cash rate target by 25 basis points to 4.35 per cent", 0.80),
    ("The Board judged an increase in interest rates was warranted to be more assured that inflation would return to target in a reasonable timeframe", 0.82),
    ("This further increase in interest rates is to provide greater confidence that inflation will return to target within a reasonable timeframe", 0.82),
    ("The Board decided to leave the cash rate target unchanged", 0.52),
    ("The Board decided to leave the cash rate target unchanged at 4.35 per cent", 0.55),
    ("The Board took the decision to hold interest rates steady this month to provide additional time to assess the impact of the increase in interest rates to date", 0.55),
    ("The Board decided that it was appropriate to hold rates steady to provide time to assess the impact of the increase in interest rates so far", 0.55),
    ("The Board decided to lower the cash rate target by 25 basis points", 0.25),
    ("At its meeting today the Board decided to lower the cash rate target to 4.10 per cent", 0.25),
    ("In removing a little of the policy restrictiveness in its decision today the Board acknowledges that progress has been made", 0.28),
    ("The Board decided to reduce the cash rate target by 50 basis points", 0.15),
    ("The Board decided to lower the cash rate target to 0.25 per cent", 0.05),
    ("The Board decided to reduce the cash rate to 0.10 per cent", 0.03),
    ("The Board expects that further increases in interest rates will be needed over the months ahead to ensure that inflation returns to target", 0.90),
    ("The Board expects that further tightening of monetary policy will be needed to ensure that inflation returns to target and that this period of high inflation is only temporary", 0.88),
    ("Some further tightening of monetary policy may well be needed to ensure that inflation returns to target", 0.82),
    ("Some further tightening of monetary policy may be required to ensure that inflation returns to target in a reasonable timeframe", 0.80),
    ("Whether further tightening of monetary policy is required will depend upon the data and the evolving assessment of risks", 0.70),
    ("The Board is not ruling anything in or out", 0.58),
    ("The Board remains vigilant to the risk of continued high inflation", 0.75),
    ("The Board is not ruling anything out", 0.60),
    ("Keeping the cash rate at the current level is important to reduce inflationary pressures", 0.62),
    ("The Board remains cautious on prospects for further policy easing", 0.62),
    ("The forecasts suggest that if monetary policy is eased too much too soon disinflation could stall", 0.68),
    ("If monetary policy is eased too much too soon disinflation could stall and inflation would settle above the midpoint of the target range", 0.70),
    ("The Board expects it will be some time before inflation is sustainably low and stable", 0.65),
    ("The Board does not see inflation returning sustainably to target until 2026", 0.65),
    ("The process of returning inflation to target has been slow and bumpy", 0.68),
    ("The Board acknowledges that progress has been made but is cautious about the outlook", 0.58),
    ("The Board is prepared to use its full range of tools to support the economy", 0.10),
    ("The Board will do what is necessary to support the Australian economy", 0.12),
    ("The Board considers it unlikely that the cash rate target will be increased for at least three years", 0.08),
    ("The Board will not increase the cash rate target until actual inflation is sustainably within the 2 to 3 per cent target range", 0.10),
    ("The Board will maintain highly accommodative policy settings as long as is required", 0.08),
    ("Some of the upside risks to inflation appear to have eased and there are signs that disinflation might be occurring a little more quickly than earlier expected", 0.30),
    ("These factors give the Board more confidence that inflation is moving sustainably towards the midpoint of the 2 to 3 per cent target range", 0.28),
    ("Inflationary pressures are easing a little more quickly than expected", 0.28),
    ("There has also been continued subdued growth in private demand and wage pressures have eased", 0.30),
    ("The Board has gained confidence that inflation is moving sustainably towards the midpoint of the target range", 0.25),
    ("Progress on inflation and the evolving balance of risks give the Board confidence to begin easing policy", 0.22),
    ("The Board will continue to rely upon the data and the evolving assessment of risks to guide its decisions", 0.50),
    ("The Board will continue to pay close attention to developments in the global economy trends in domestic demand and the outlook for inflation and the labour market", 0.50),
    ("The path of interest rates that will best ensure that inflation returns to target in a reasonable timeframe will depend upon the data and the evolving assessment of risks", 0.50),
    ("The Board will be paying close attention to developments in the global economy trends in household spending and the outlook for inflation and the labour market", 0.50),
    ("How interest rates evolve from here is uncertain and the Board is watching developments closely", 0.52),
    ("Inflation in Australia is too high", 0.88),
    ("Inflation at 7 per cent is still too high and it will be some time yet before it is back in the target range", 0.90),
    ("Inflation is still too high and is proving more persistent than expected", 0.88),
    ("CPI inflation was 7.8 per cent the highest since 1990", 0.92),
    ("In underlying terms inflation was 6.9 per cent which was higher than expected", 0.90),
    ("Inflation expectations risk becoming unanchored", 0.92),
    ("If high inflation were to become entrenched in peoples expectations it would be much more costly to reduce later", 0.85),
    ("There is a risk that expectations of ongoing high inflation contribute to larger increases in both prices and wages", 0.85),
    ("Inflation is still some way above the midpoint of the 2 to 3 per cent target range", 0.72),
    ("Inflation is proving persistent", 0.75),
    ("Underlying inflation has been above the midpoint of the target for 11 consecutive quarters", 0.78),
    ("Quarterly underlying CPI inflation has fallen very little over the past year", 0.75),
    ("Services price inflation remains high with strong demand for some services", 0.72),
    ("Rents are increasing at the fastest rate in some years with vacancy rates low in many parts of the country", 0.70),
    ("The prices of many services are continuing to rise briskly", 0.72),
    ("Inflation has been higher than expected and labour market conditions have proven stronger than anticipated", 0.70),
    ("The recent data have demonstrated that the process of returning inflation to target has been slow and bumpy", 0.68),
    ("Services price inflation remains elevated in many economies", 0.65),
    ("Goods price inflation has eased but services price inflation remains high", 0.62),
    ("Inflation has fallen substantially since the peak in 2022", 0.25),
    ("Inflation has come down considerably from its peak", 0.22),
    ("The disinflationary process is continuing", 0.22),
    ("Underlying inflation was 3.2 per cent which suggests inflationary pressures are easing a little more quickly than expected", 0.28),
    ("Inflation has passed its peak but is still too high", 0.45),
    ("Inflation continues to moderate", 0.32),
    ("A range of information suggests that inflation has peaked in Australia", 0.35),
    ("The monthly CPI indicator suggests that inflation has peaked in Australia", 0.35),
    ("Goods price inflation has declined but services price inflation remains high", 0.42),
    ("Goods price inflation is expected to moderate over the months ahead", 0.35),
    ("Medium-term inflation expectations remain well anchored and it is important that this remains the case", 0.40),
    ("Longer term inflation expectations have been consistent with the inflation target", 0.38),
    ("Overall measures of inflation expectations remain consistent with the inflation target", 0.40),
    ("Inflation has eased substantially while the labor market has remained strong", 0.32),
    ("The decline in inflation reflects the easing of supply-demand imbalances", 0.30),
    ("The labour market remains very tight", 0.80),
    ("The labour market remains extremely tight with the unemployment rate at historically low levels", 0.82),
    ("Conditions in the labour market remain tight", 0.72),
    ("The unemployment rate is at a near 50-year low", 0.78),
    ("Labour market conditions remain tight and in fact tightened a little further", 0.75),
    ("Many firms continue to experience difficulty hiring workers", 0.72),
    ("Job vacancies and job ads are both at very high levels", 0.75),
    ("Demand has remained above the economys capacity to supply goods and services", 0.78),
    ("The availability of labour is still a constraint for a range of employers", 0.70),
    ("Measures of labour underutilisation have declined", 0.72),
    ("Productivity growth has not picked up which implies that growth in unit labour costs remains high", 0.75),
    ("Unit labour costs are rising briskly with productivity growth remaining subdued", 0.78),
    ("The Board remains alert to the risk of a prices-wages spiral given the limited spare capacity in the economy", 0.82),
    ("Labour market conditions are easing but are still tight relative to full employment", 0.42),
    ("Conditions in the labour market have eased but remain tight", 0.40),
    ("Labour market conditions have eased", 0.32),
    ("The unemployment rate has increased", 0.25),
    ("The unemployment rate is expected to increase", 0.30),
    ("Firms report that labour shortages have eased", 0.35),
    ("The number of vacancies has declined", 0.35),
    ("Wage pressures have eased a little more than expected", 0.30),
    ("Wage pressures have eased", 0.32),
    ("Wages growth is not expected to increase much further", 0.38),
    ("The gap between jobs and workers has narrowed", 0.32),
    ("There is a risk of a sharper deterioration in the labour market than currently projected", 0.22),
    ("The economy has been stronger than expected", 0.72),
    ("The Australian economy grew strongly", 0.70),
    ("GDP growth has been above trend", 0.68),
    ("Economic activity has been surprisingly resilient", 0.68),
    ("Household spending remains strong supported by a solid labour market", 0.68),
    ("Consumer spending has been surprisingly resilient", 0.68),
    ("The economy is operating at a high level of capacity utilisation", 0.65),
    ("Demand continues to exceed supply in the economy", 0.75),
    ("Growth in output has been weak", 0.25),
    ("Private domestic demand is recovering a little more slowly than earlier expected", 0.28),
    ("Economic growth remains slow", 0.25),
    ("Higher interest rates have led people to cut back on spending", 0.30),
    ("The economy has been experiencing a period of below-trend growth", 0.28),
    ("Household consumption growth is weak as is dwelling investment", 0.22),
    ("High inflation is weighing on peoples real incomes and household consumption growth is weak", 0.25),
    ("The combination of higher interest rates and cost-of-living pressures is leading to a substantial slowing in household spending", 0.25),
    ("GDP growth is expected to slow", 0.30),
    ("Growth is expected to remain low over the next year", 0.28),
    ("The pace of economic growth has slowed significantly", 0.20),
    ("Consumer spending has weakened reflecting the squeeze on household budgets", 0.22),
    ("The economy contracted modestly", 0.15),
    ("Monetary policy has been restrictive and will remain so", 0.48),
    ("The Boards assessment is that monetary policy has been restrictive and will remain so after this reduction in the cash rate", 0.35),
    ("Higher interest rates are working to establish a more sustainable balance between supply and demand in the economy", 0.48),
    ("Higher interest rates are working to bring aggregate demand and supply closer towards balance", 0.45),
    ("The current stance of monetary policy is restrictive", 0.50),
    ("The full effect of the cumulative increase in interest rates is yet to be felt", 0.55),
    ("Housing prices are rising again", 0.62),
    ("Housing prices are continuing to rise across the country", 0.62),
    ("Housing cost inflation is abating", 0.35),
    ("Dwelling investment remains weak", 0.30),
    ("Rents are increasing at the fastest rate in some years", 0.70),
    ("Global inflation remains very high", 0.65),
    ("Most central banks have been easing monetary policy as they become more confident that inflation is moving sustainably back towards their targets", 0.35),
    ("The outlook for the global economy remains subdued with below average growth expected", 0.30),
    ("Upside risks to inflation remain", 0.78),
    ("Services price inflation has been surprisingly persistent overseas and the same could occur in Australia", 0.72),
    ("There is a risk that high inflation could become entrenched in peoples expectations", 0.85),
    ("The Board remains alert to the risk of a prices-wages spiral", 0.80),
    ("There is a risk of continued subdued output growth and a sharper deterioration in the labour market", 0.22),
    ("Downside risks to the global outlook have increased", 0.25),
    ("There are risks on both sides", 0.45),
    ("There are notable uncertainties about the outlook for domestic economic activity and inflation", 0.45),
    ("The economic outlook is uncertain", 0.45),
    ("The path to achieving a soft landing remains a narrow one", 0.52),
    ("Sustainably returning inflation to target within a reasonable timeframe remains the Boards highest priority", 0.60),
    ("Returning inflation to target within a reasonable timeframe remains the Boards priority", 0.60),
    ("The Boards priority is to return inflation to target", 0.62),
    ("Wages growth is continuing to pick up in response to the tight labour market and higher inflation", 0.72),
    ("Wages growth has picked up but at the aggregate level remains consistent with the inflation target provided productivity growth picks up", 0.55),
    ("Wage growth has shown signs of moderating", 0.35),
    ("Wage pressures have eased more than expected", 0.28),
    ("This is consistent with the RBAs mandate for price stability and full employment", 0.50),
    ("The Board will continue to monitor the implications of incoming data for the economic outlook", 0.50),
    ("The Board is seeking to return inflation to the 2 to 3 per cent target range while keeping the economy on an even keel", 0.50),
    ("Inflation has passed its peak but remains above target", 0.48),
    ("The labour market remains tight though conditions have eased somewhat", 0.48),
    ("Economic growth has slowed but remains positive", 0.42),
    ("Higher interest rates are working as intended but inflation remains above target", 0.52),
    ("The economy is adjusting to higher interest rates with demand slowing and inflation moderating", 0.45),
    ("Progress on inflation has been welcome but the job is not yet done", 0.55),
    ("Labour market conditions are easing gradually but remain tight", 0.45),
    ("While today's decision recognises the welcome progress on inflation the Board remains cautious", 0.45),
]

# ── Training data — UK ────────────────────────────────────────────────────────
_TRAINING_UK = [
    ("The MPC voted to increase Bank Rate by 0.75 percentage points", 0.95),
    ("The MPC voted by a majority to increase Bank Rate by 0.5 percentage points", 0.88),
    ("The MPC voted by a majority to increase Bank Rate by 0.25 percentage points", 0.78),
    ("The MPC voted unanimously to increase Bank Rate by 0.25 percentage points", 0.80),
    ("The MPC voted by a majority to maintain Bank Rate at 5.25 per cent", 0.55),
    ("The MPC voted to maintain Bank Rate", 0.52),
    ("The MPC voted by a majority of 6-3 to maintain Bank Rate at 4.75 per cent", 0.55),
    ("The MPC voted by a majority to reduce Bank Rate by 0.25 percentage points", 0.25),
    ("The MPC voted to reduce Bank Rate by 0.5 percentage points", 0.15),
    ("The MPC voted to reduce Bank Rate to 0.1 per cent", 0.05),
    ("At its meeting the MPC voted by a majority of 5-4 to reduce Bank Rate by 0.25 percentage points", 0.28),
    ("Four members preferred to increase Bank Rate by 0.25 percentage points", 0.58),
    ("Two members preferred to increase Bank Rate by 0.5 percentage points", 0.60),
    ("Three members voted to maintain Bank Rate", 0.42),
    ("The MPC voted by a majority of 5-4 to maintain Bank Rate with four members preferring to increase", 0.58),
    ("Two members preferred to maintain Bank Rate", 0.45),
    ("Three members preferred to reduce Bank Rate by 0.25 percentage points", 0.42),
    ("Two members preferred to reduce Bank Rate by 0.25 percentage points", 0.45),
    ("Four members voted to maintain Bank Rate at 4 per cent", 0.55),
    ("Two members voted to reduce Bank Rate by 0.25 percentage points", 0.45),
    ("If there were to be evidence of more persistent pressures then further tightening in monetary policy would be required", 0.85),
    ("The MPC will adjust Bank Rate as necessary to return inflation to the 2 per cent target sustainably in the medium term", 0.60),
    ("Monetary policy will need to be sufficiently restrictive for sufficiently long to return inflation to the 2 per cent target sustainably", 0.78),
    ("The MPC will ensure that Bank Rate is sufficiently restrictive for sufficiently long to return inflation to the 2 per cent target", 0.75),
    ("Interest rates need to remain high enough to be confident that inflation will fall all the way to target and stay there", 0.72),
    ("The MPC will continue to monitor closely indications of persistent inflationary pressures", 0.68),
    ("The Committee remains focused on squeezing out any existing or emerging persistent inflationary pressures", 0.72),
    ("Domestic inflationary pressures are resolving more slowly", 0.68),
    ("Remaining domestic inflationary pressures are resolving more slowly than expected", 0.72),
    ("The risks from greater inflation persistence remain", 0.70),
    ("The MPC will need to see further evidence that inflationary pressures are easing", 0.65),
    ("The Committee judges that monetary policy needs to remain restrictive", 0.68),
    ("The Bank of England stands ready to take whatever additional action is necessary", 0.10),
    ("The MPC will take whatever action is needed to achieve its remit", 0.12),
    ("The Committee continues to judge that some degree of monetary policy easing is warranted", 0.22),
    ("There has been substantial disinflation over the past two and a half years supported by the restrictive stance of monetary policy", 0.30),
    ("That progress has allowed for reductions in Bank Rate over the past year", 0.28),
    ("Bank Rate is likely to continue on a gradual downward path", 0.25),
    ("On the basis of the current evidence Bank Rate is likely to continue on a gradual downward path", 0.28),
    ("The restrictiveness of policy has fallen as Bank Rate has been reduced", 0.30),
    ("The risk from greater inflation persistence has become somewhat less pronounced", 0.32),
    ("The MPC's approach is to reduce the degree of monetary policy restrictiveness gradually", 0.30),
    ("The Committee judges that some reduction in the degree of policy restrictiveness is now appropriate", 0.25),
    ("A gradual approach to removing monetary policy restrictiveness remains appropriate", 0.42),
    ("A gradual approach allows the MPC to assess carefully the balance of risks to inflation as the evidence evolves", 0.45),
    ("The extent of further easing in monetary policy will depend on the evolution of the outlook for inflation", 0.45),
    ("The MPC will continue to decide the appropriate level of Bank Rate at each meeting", 0.50),
    ("Monetary policy has been guided by the need to squeeze remaining inflationary pressures out of the economy", 0.60),
    ("CPI inflation remains well above the 2 per cent target", 0.85),
    ("Inflation is still too high", 0.85),
    ("CPI inflation was 10.5 per cent its highest level in over 40 years", 0.95),
    ("Global consumer price inflation remains high", 0.75),
    ("UK domestic inflationary pressures have been firmer than expected", 0.80),
    ("Both private sector regular pay growth and services CPI inflation have been notably higher than forecast", 0.82),
    ("Services CPI inflation remains elevated", 0.72),
    ("Services consumer price inflation remains persistent", 0.72),
    ("Core CPI inflation remains elevated", 0.72),
    ("Pay growth remains elevated and above rates consistent with the 2 per cent target", 0.75),
    ("Annual private sector regular average weekly earnings growth remains above target-consistent levels", 0.72),
    ("Second-round effects in domestic prices and wages have taken longer to unwind than they did to emerge", 0.70),
    ("Indicators of inflation persistence remain elevated", 0.70),
    ("Services price inflation is expected to remain broadly unchanged in the near term", 0.62),
    ("Forward-looking wage indicators have remained elevated", 0.68),
    ("CPI inflation has fallen back sharply", 0.22),
    ("There has been substantial disinflation", 0.22),
    ("CPI inflation has fallen to the 2 per cent target", 0.20),
    ("Inflation has returned to around the 2 per cent target", 0.20),
    ("Underlying disinflation has generally continued", 0.28),
    ("Twelve-month CPI inflation has fallen", 0.32),
    ("CPI inflation is expected to fall significantly", 0.28),
    ("CPI inflation is projected to return to the 2 per cent target", 0.30),
    ("Reflecting restrictive monetary policy pay growth and services price inflation have continued to ease", 0.30),
    ("Pay growth has eased", 0.32),
    ("Private sector regular AWE pay growth has fallen", 0.32),
    ("A range of pay settlements data suggest that pay growth is moderating", 0.35),
    ("Headline CPI inflation is expected to decline significantly", 0.28),
    ("The decline in headline inflation largely reflects the easing of external price pressures", 0.30),
    ("The labour market remains tight", 0.72),
    ("Wage growth has been stronger than can be explained by economic fundamentals", 0.78),
    ("Pay growth has been above rates consistent with meeting the inflation target", 0.75),
    ("There has been upside news on wages and the labour market", 0.72),
    ("The Committee now judges that the labour market is broadly in balance", 0.38),
    ("The labour market has continued to loosen", 0.30),
    ("The labour market had loosened further", 0.28),
    ("The unemployment rate has continued to move higher", 0.25),
    ("The LFS unemployment rate has risen", 0.28),
    ("The redundancy rate has risen to its highest level since 2013", 0.22),
    ("Labour market tightness has eased", 0.32),
    ("There is evidence of subdued economic growth and building slack in the labour market", 0.25),
    ("UK GDP growth has been stronger than expected", 0.68),
    ("GDP growth has surprised to the upside", 0.68),
    ("Consumer spending has been more resilient than anticipated", 0.65),
    ("GDP growth has been subdued", 0.28),
    ("Most indicators of UK near-term activity have declined", 0.25),
    ("GDP growth is expected to have been weaker than projected", 0.25),
    ("A margin of economic slack is projected to emerge", 0.25),
    ("The economy has been experiencing a period of below-trend growth", 0.28),
    ("Monthly GDP declined", 0.22),
    ("The economy is in or close to recession", 0.15),
    ("Consumer spending has weakened reflecting the squeeze on real incomes", 0.25),
    ("The Committee voted unanimously to reduce the stock of UK government bond purchases by 100 billion over the next twelve months", 0.68),
    ("The Bank should reduce the stock of UK government bond purchases held for monetary policy purposes", 0.65),
    ("The MPC stands ready to make further asset purchases if needed", 0.15),
    ("The Committee voted to increase the stock of UK government bond purchases", 0.10),
    ("Gilt yields have risen materially particularly at shorter maturities", 0.58),
    ("Financial conditions have tightened", 0.48),
    ("Credit conditions have tightened and are weighing on economic activity", 0.35),
    ("Global activity indicators have been more resilient than expected", 0.58),
    ("There has been a further increase in geopolitical and global trade policy uncertainty", 0.38),
    ("Trade policy uncertainty may weigh on activity in a number of advanced economies including the United Kingdom", 0.32),
    ("There are upside risks to the inflation outlook from domestic price and wage pressures", 0.78),
    ("The risks to the inflation outlook are skewed to the upside", 0.80),
    ("There remains a risk that inflation persistence could be more enduring", 0.75),
    ("Second-round effects in wages and prices pose upside risks to inflation", 0.75),
    ("The risk to medium-term inflation from weaker demand remains", 0.28),
    ("Downside risks to the growth outlook have increased", 0.22),
    ("The risks around achieving the inflation target are more balanced", 0.38),
    ("Monetary policy is being set to ensure CPI inflation settles sustainably at 2 per cent in the medium term which involves balancing the risks", 0.45),
    ("Wholesale gas futures and oil prices have fallen materially", 0.30),
    ("Energy prices have risen sharply contributing to higher inflation", 0.78),
    ("As the effects of the energy price shock unwound headline CPI inflation has fallen sharply", 0.25),
    ("Announced increases in the National Living Wage and employer National Insurance contributions", 0.62),
    ("Firms have indicated that they expect to pass on some of the increase in costs from higher employer NICs", 0.65),
    ("The MPC sets monetary policy to meet the 2 per cent inflation target and in a way that helps to sustain growth and employment", 0.50),
    ("The MPC adopts a medium-term and forward-looking approach to determine the monetary stance", 0.50),
    ("There are signs that the persistence of inflation may be fading", 0.32),
    ("Key indicators of inflation persistence have remained elevated", 0.72),
    ("Second-round effects in domestic prices and wages are taking longer to unwind", 0.70),
    ("The MPC judges that the risks to medium-term inflation from greater inflation persistence and weaker demand are more balanced", 0.40),
    ("Underlying inflation persistence has eased but remains above target-consistent levels", 0.55),
    ("CPI inflation has fallen but remains above the 2 per cent target", 0.48),
    ("Pay growth has eased but remains elevated relative to target-consistent levels", 0.52),
    ("The labour market has loosened but conditions remain relatively tight by historical standards", 0.45),
    ("Growth has been subdued but the economy has avoided recession", 0.42),
    ("Inflation has fallen significantly from its peak but domestic price pressures remain", 0.48),
    ("There are encouraging signs of disinflation but services inflation remains sticky", 0.52),
    ("The economy is adjusting to the higher level of interest rates with demand slowing and inflation moderating", 0.45),
    ("GDP growth has picked up somewhat but remains below its potential rate", 0.42),
    ("The recent data on inflation have been mixed with goods prices easing but services prices remaining firm", 0.52),
    ("Labour market indicators present a mixed picture with some signs of loosening alongside continued tightness in certain sectors", 0.48),
]

# ── Dataset helper ────────────────────────────────────────────────────────────
class _SentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Shared tokenizer ──────────────────────────────────────────────────────────
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Training helper ───────────────────────────────────────────────────────────
def _train_model(training_data, label: str, epochs: int):
    print(f"Training {label} model ({len(training_data)} examples, {epochs} epochs)...")

    texts  = [s[0] for s in training_data]
    scores = [s[1] for s in training_data]

    encodings = _tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=_MAX_LEN,
        return_tensors="pt",
    )
    labels  = torch.tensor(scores, dtype=torch.float32)
    dataset = _SentDataset(encodings, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    ).to(device)

    args = TrainingArguments(
        output_dir=f"./models/{label.lower()}",
        num_train_epochs=epochs,
        per_device_train_batch_size=_BATCH,
        learning_rate=_LR,
        logging_steps=50,
        save_strategy="no",
        use_cpu=True,
    )
    Trainer(model=model, args=args, train_dataset=dataset).train()
    model.eval()
    print(f"  [{label}] training complete.")
    return model


# ── Train all three models at import time ─────────────────────────────────────
print("Initializing analyze.py — training US / AU / UK models...")
_mod_us = _train_model(_TRAINING_US, "US", _US_EPOCHS)
_mod_au = _train_model(_TRAINING_AU, "AU", _AU_EPOCHS)
_mod_uk = _train_model(_TRAINING_UK, "UK", _UK_EPOCHS)
print("All models ready.")


# ── Country routing ───────────────────────────────────────────────────────────
def _resolve_country(country: str):
    """Return (key, model, skip_list) for the given country string."""
    c = country.strip()
    if c in ("US", "United States", "Pakistan"):
        return "US", _mod_us, SKIP_US
    if c == "Australia":
        return "AU", _mod_au, SKIP_AU
    if c in ("UK", "United Kingdom"):
        return "UK", _mod_uk, SKIP_UK
    return "US", _mod_us, SKIP_US


# ── Sentence scoring ──────────────────────────────────────────────────────────
def _score_sentence(text: str, model) -> float:
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=_MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = float(model(**inputs).logits.cpu().numpy()[0][0])
    return float(np.clip(raw, 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_statement(country: str, statement: str) -> dict:
    """
    Score every sentence in `statement` using the appropriate country model
    and predict the 60-day yield direction from sentiment.

    Parameters
    ----------
    country   : "US", "United States", "Pakistan", "Australia",
                "UK", or "United Kingdom"
    statement : raw central-bank statement text

    Returns
    -------
    dict with keys: avg_score, hawk_count, dove_count, neutral_count,
                    total_sentences, scored_sentences, prediction
    """
    key, model, skip_list = _resolve_country(country)
    reg = _REG[key]

    scored_sentences = []
    for sent in [s.strip() for s in statement.split(".")]:
        if len(sent) < 20:
            continue
        if any(pat in sent for pat in skip_list):
            continue
        score = _score_sentence(sent, model)
        scored_sentences.append({"text": sent, "score": round(score, 4)})

    if not scored_sentences:
        avg_score = 0.5
    else:
        avg_score = float(np.mean([s["score"] for s in scored_sentences]))

    hawk_count    = sum(1 for s in scored_sentences if s["score"] > 0.55)
    dove_count    = sum(1 for s in scored_sentences if s["score"] < 0.45)
    neutral_count = len(scored_sentences) - hawk_count - dove_count

    demeaned = avg_score - reg["mean_sentiment"]
    change   = reg["intercept"] + reg["coef_sentiment"] * demeaned
    margin   = _CI_MULTIPLIER * reg["residual_std"]
    prediction = {
        "change":    round(change, 4),
        "lower":     round(change - margin, 4),
        "upper":     round(change + margin, 4),
        "direction": "RISING" if change > 0 else "FALLING",
    }

    return {
        "avg_score":        round(avg_score, 4),
        "hawk_count":       hawk_count,
        "dove_count":       dove_count,
        "neutral_count":    neutral_count,
        "total_sentences":  len(scored_sentences),
        "scored_sentences": scored_sentences,
        "prediction":       prediction,
    }
