# Train/Test Contamination Measurement (local, independent run)

- FinGPT/fingpt-sentiment-train rows: **76772**  (unique normalized inputs: 30209)
- FPB sentences_allagree rows: **2259**  | class distribution: {'neutral': 1386, 'positive': 570, 'negative': 303}

## Overlap of the FPB evaluation set inside the FinGPT training corpus
- **Exact verbatim match:** 1699/2259 = **75.21%**
- **Normalized match** (lowercase/strip-punct/collapse-ws): 1699/2259 = **75.21%**
- **Label agreement among matched:** 1699/1699 = **100.00%** (matched sentences carry the SAME gold label in training)
- **Decontaminated remainder** (FPB sentences NOT in training): **560** (24.79%) | class dist: {'neutral': 353, 'negative': 77, 'positive': 130}

## Interpretation
The model is fine-tuned on FinGPT/fingpt-sentiment-train and evaluated on FPB. 75.2% of the evaluation sentences — with identical gold labels — were in the training pool. The reported 99.56% is therefore largely memorization, not generalization. A valid number must be reported on the 560-sentence decontaminated remainder and on an external dataset never present in fingpt-sentiment-train.

## Sample matched (leaked) sentences
- [neutral] The equipment will be made at Vaahto 's plant in Hollola in Finland , and delivery is scheduled for the first quarter of 2009 .
- [positive] 27 January 2011 - Finnish IT solutions provider Affecto Oyj ( HEL : AFE1V ) said today it has won a EUR1 .2 m ( USD1 .6 m ) contract from st
- [positive] ( ADP News ) - Sep 30 , 2008 - Finnish security and privacy software solutions developer Stonesoft Oyj said today that it won a USD 1.9 mill
- [neutral] Also , a six-year historic analysis is provided for this market .
- [neutral] Estonia 's Agriculture Minister Helir-Valdor Seeder is in Finland on a two-day visit , in the course of which he will meet with his Finnish 
- [neutral] According to HK Ruokatalo , almost all the meat used by the company comes from Finland .
- [negative] Cargo traffic fell 1 % year-on-year to 8,561 tonnes in September 2009 .
- [neutral] With the U.S. Federal Government putting a stake in the ground , vendors - and their customers - are focused on meeting the deadline .
- [negative] Repeats sees 2008 operating profit down y-y ( Reporting by Helsinki Newsroom ) Keywords : TECNOMEN-RESULTS
- [neutral] Componenta is a metal sector company with international operations and production plants located in Finland , the Netherlands , Sweden and T
- [neutral] The company 's board of directors will propose a dividend of EUR 0.95 per share for 2008 at the annual general meeting , scheduled to be hel
- [neutral] Indigo and Somoncom serve 377,000 subscribers and had a market share of approximately 27 % as of May 2007 .
- [neutral] Rautalinko was resposnible also for Mobility Services , and his job in this division will be continued by Marek Hintze .
- [positive] Revenues at the same time grew 14 percent to 43 million euros .
- [positive] Department store sales improved by 14 % to EUR 1,070.6 mn .
