# Citation insertion map for the three-scenario manuscript

Generated on 2026-06-11.

Purpose: lock paragraph-level citation placement for `docs/transport_recalibration_three_scenario_manuscript_draft.md` and record DOI status for the added references.

## Summary

The manuscript should use references to support four kinds of claims:

- prediction-model reporting, risk-of-bias, and external validation standards;
- clinical AI transportability, dataset shift, and real-world deployment failures;
- calibration, model updating, recalibration, DCA, and bootstrap methodology;
- data-resource and disease-context citations for NHANES, MIMIC-IV, eICU, and diabetes mortality.

Two items in the proposed map need correction:

- I could not verify a `van Smeden et al. 2019, Journal of Clinical Epidemiology` paper specifically about prediction-model external validation. Use Debray 2017 BMJ and Collins 2014 BMC Medical Research Methodology for that claim. If a van Smeden citation is still desired, the verified JCE item is `van Smeden et al. 2021, Clinical prediction models: diagnosis versus prognosis`, DOI `10.1016/j.jclinepi.2021.01.009`, but it does not specifically cover external validation methods.
- `Finlayson 2021` on dataset shift is an NEJM Correspondence, DOI `10.1056/NEJMc2104626`. The Science paper by Finlayson is the 2019 adversarial-attacks article, DOI `10.1126/science.aaw4399`.

## Paragraph-level insertion map

| Manuscript location | Current marker | Claim being supported | Recommended references |
|---|---:|---|---|
| Introduction paragraph 1, sentence on reporting and evaluation guidance | `[1,2]` | Reporting, risk of bias, and model-study appraisal are core to clinical prediction model research. | Collins et al. 2024 TRIPOD+AI; Moons et al. 2025 PROBAST+AI. Add Wolff et al. 2019 PROBAST if discussing original PROBAST or risk of bias in detail. |
| Introduction paragraph 1, sentence on EHR/IPD heterogeneity | `[3-6]` | External validation using EHR/IPD data exposes heterogeneity hidden by pooled discrimination. | Riley et al. 2016 BMJ; Debray et al. 2017 BMJ; Collins et al. 2014 BMC Med Res Methodol; Steyerberg et al. 2010 Epidemiology. |
| Introduction paragraph 2, Epic sepsis external validation | `[7]` | Real-world clinical AI models can underperform under external validation. | Wong et al. 2021 JAMA Internal Medicine. |
| Introduction paragraph 2, calibration drift in EHR models | `[8,9]` | EHR-derived AKI and hospital mortality models show temporal calibration drift. | Davis et al. 2017 JAMIA; Davis et al. 2017 AMIA Annual Symposium Proceedings. |
| Introduction paragraph 2, cross-site evaluations and deployment failure | `[10-14]` | Models can learn site-specific, temporal, or measurement-process patterns that do not transport. | Nestor et al. 2019 PMLR; Wiens et al. 2019 Nature Medicine; Futoma et al. 2020 Lancet Digital Health; Lasko et al. 2024 npj Digital Medicine; Finlayson et al. 2021 NEJM or Finlayson et al. 2019 Science depending on whether the sentence emphasizes dataset shift or adversarial/gaming risk. |
| Introduction paragraph 3, calibration reporting and calibration hierarchy | `[15-19]` | Calibration-in-the-large, slope, and calibration curves should accompany discrimination. | Van Calster et al. 2019 BMC Medicine; Van Calster et al. 2016 JCE; Steyerberg et al. 2010 Epidemiology; Riley et al. 2021 Statistics in Medicine; Collins et al. 2024 TRIPOD+AI. |
| Introduction paragraph 3, recalibration methods | `[20-25]` | Intercept-only, logistic/Platt, model updating, and isotonic recalibration are established updating strategies. | Steyerberg et al. 2004 Statistics in Medicine; Janssen et al. 2008 JCE; Platt 1999; Zadrozny and Elkan 2002 KDD; Van Calster et al. 2023 BMC Medicine; Nieboer et al. 2016 BMC Med Res Methodol or Vergouwe et al. 2017 Statistics in Medicine. |
| Introduction paragraph 4, diabetes mortality context | `[26,27]` | Diabetes is common and associated with excess mortality and cardiovascular risk. | Seshasai et al. 2011 NEJM; GBD 2021 Diabetes Collaborators 2023 Lancet. For a GBD 2019 mortality-specific option, use GBD 2019 Diabetes Mortality Collaborators 2022 Lancet Diabetes Endocrinology. |
| Discussion paragraph 3 | `[15-18]`, `[8,9]` | Discrimination and calibration are complementary; calibration can drift while discrimination persists. | Reuse Van Calster 2019, Van Calster 2016, Steyerberg 2010, Davis 2017 JAMIA, Davis 2017 AMIA. |
| Discussion paragraph 4 | `[28]` | Sample size and event-count needs for validation/recalibration depend on precision targets. | Riley et al. 2021 Statistics in Medicine; optionally Snell et al. 2021 JCE for simulation-based external-validation sample size. |
| Methods, model development and reporting | Add citations | Reporting guidance and software implementation. | Collins et al. 2024 TRIPOD+AI; Pedregosa et al. 2011 JMLR for scikit-learn. |
| Methods, source-shift/source classifier | Add citations | Dataset shift and cross-site deployment diagnostic rationale. | Finlayson et al. 2021 NEJM; Lasko et al. 2024 npj Digital Medicine; Nestor et al. 2019 PMLR. |
| Methods, calibration regression and ECE | Add citations | Calibration slope/intercept and risk-model calibration. | Cox 1958 Biometrika; Van Calster et al. 2016 JCE; Van Calster et al. 2019 BMC Medicine. |
| Methods, event-count recalibration | Add citations | Updating/recalibration methods. | Steyerberg et al. 2004; Janssen et al. 2008; Platt 1999; Zadrozny and Elkan 2002. |
| Methods, DCA | Add citations | Decision-curve analysis and calibration impact on utility. | Vickers and Elkin 2006; Van Calster and Vickers 2015. |
| Methods, uncertainty | Add citations | Bootstrap confidence intervals. | Efron 1979. |
| Data sources | Add citations | Data resource provenance. | NCHS NHANES; NCHS linked mortality; Johnson et al. 2023 Scientific Data; Johnson et al. 2024 PhysioNet MIMIC-IV v3.1; Pollard et al. 2018 Scientific Data; Pollard et al. 2019 PhysioNet eICU v2.0. |

## DOI-verified reference worklist

### Reporting, risk of bias, and external validation

1. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. DOI: `10.1136/bmj-2023-078378`.
2. Moons KGM, Damen JAAG, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ*. 2025;388:e082505. DOI: `10.1136/bmj-2024-082505`.
3. Wolff RF, Moons KGM, Riley RD, Whiting PF, Westwood M, Collins GS, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. *Ann Intern Med*. 2019;170:51-58. DOI: `10.7326/M18-1376`.
4. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). *Ann Intern Med*. 2015;162:55-63. DOI: `10.7326/M14-0697`.
5. Riley RD, Ensor J, Snell KIE, Debray TPA, Altman DG, Moons KGM, Collins GS. External validation of clinical prediction models using big datasets from e-health records or IPD meta-analysis. *BMJ*. 2016;353:i3140. DOI: `10.1136/bmj.i3140`.
6. Debray TPA, Damen JAAG, Snell KIE, Ensor J, Hooft L, Reitsma JB, Riley RD, Moons KGM. A guide to systematic review and meta-analysis of prediction model performance. *BMJ*. 2017;356:i6460. DOI: `10.1136/bmj.i6460`.
7. Collins GS, de Groot JA, Dutton S, et al. External validation of multivariable prediction models: a systematic review of methodological conduct and reporting. *BMC Med Res Methodol*. 2014;14:40. DOI: `10.1186/1471-2288-14-40`.
8. Steyerberg EW, Vickers AJ, Cook NR, Gerds T, Gonen M, Obuchowski N, Pencina MJ, Kattan MW. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*. 2010;21:128-138. DOI: `10.1097/EDE.0b013e3181c30fb2`.

### Clinical AI transport, dataset shift, and deployment failures

9. Wong A, Otles E, Donnelly JP, Krumm A, McCullough J, DeTroyer-Cooley O, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Intern Med*. 2021;181:1065-1070. DOI: `10.1001/jamainternmed.2021.2626`.
10. Davis SE, Lasko TA, Chen G, Siew ED, Matheny ME. Calibration drift in regression and machine learning models for acute kidney injury. *J Am Med Inform Assoc*. 2017;24:1052-1061. DOI: `10.1093/jamia/ocx030`.
11. Davis SE, Lasko TA, Chen G, Matheny ME. Calibration drift among regression and machine learning models for hospital mortality. *AMIA Annu Symp Proc*. 2017;2017:625-634. DOI: none indexed in the sources checked.
12. Nestor B, McDermott MBA, Boag W, Berner G, Naumann T, Hughes MC, Goldenberg A, Ghassemi M. Feature robustness in non-stationary health records: caveats to deployable model performance in common clinical machine learning tasks. *Proc Machine Learning for Healthcare*. 2019;PMLR 106:381-405. DOI: none listed by PMLR; use URL `https://proceedings.mlr.press/v106/nestor19a.html`.
13. Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible machine learning for health care. *Nat Med*. 2019;25:1337-1340. DOI: `10.1038/s41591-019-0548-6`.
14. Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth of generalisability in clinical research and machine learning in health care. *Lancet Digit Health*. 2020;2:e489-e492. DOI: `10.1016/S2589-7500(20)30186-2`.
15. Lasko TA, Strobl EV, Stead WW. Why do probabilistic clinical models fail to transport between sites. *npj Digit Med*. 2024;7:53. DOI: `10.1038/s41746-024-01037-4`.
16. Finlayson SG, Subbaswamy A, Singh K, Bowers J, Kupke A, Zittrain J, Kohane IS, Saria S. The clinician and dataset shift in artificial intelligence. *N Engl J Med*. 2021;385:283-286. DOI: `10.1056/NEJMc2104626`.
17. Finlayson SG, Bowers JD, Ito J, Zittrain JL, Beam AL, Kohane IS. Adversarial attacks on medical machine learning. *Science*. 2019;363:1287-1289. DOI: `10.1126/science.aaw4399`.

### Calibration, updating, and recalibration methods

18. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration hierarchy for risk models was defined: from utopia to empirical data. *J Clin Epidemiol*. 2016;74:167-176. DOI: `10.1016/j.jclinepi.2015.12.005`.
19. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17:230. DOI: `10.1186/s12916-019-1466-7`.
20. Cox DR. Two further applications of a model for binary regression. *Biometrika*. 1958;45:562-565. DOI: `10.1093/biomet/45.3-4.562`.
21. Steyerberg EW, Borsboom GJJM, van Houwelingen HC, Eijkemans MJC, Habbema JDF. Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. *Stat Med*. 2004;23:2567-2586. DOI: `10.1002/sim.1844`.
22. Janssen KJM, Moons KGM, Kalkman CJ, Grobbee DE, Vergouwe Y. Updating methods improved the performance of a clinical prediction model in new patients. *J Clin Epidemiol*. 2008;61:76-86. DOI: `10.1016/j.jclinepi.2007.04.018`.
23. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers*. MIT Press; 1999. DOI: none.
24. Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. *KDD*. 2002:694-699. DOI: `10.1145/775047.775151`.
25. Van Calster B, Steyerberg EW, Wynants L, van Smeden M. There is no such thing as a validated prediction model. *BMC Med*. 2023;21:70. DOI: `10.1186/s12916-023-02779-w`.
26. Nieboer D, Vergouwe Y, Ankerst DP, Roobol-Bouts M, Steyerberg EW. Improving prediction models with new markers: a comparison of updating strategies. *BMC Med Res Methodol*. 2016;16:128. DOI: `10.1186/s12874-016-0231-2`.
27. Vergouwe Y, Nieboer D, Oostenbrink R, Debray TPA, Murray GD, Kattan MW, et al. A closed testing procedure to select an appropriate method for updating prediction models. *Stat Med*. 2017;36:4529-4539. DOI: `10.1002/sim.7179`.

### Decision utility, sample size, and uncertainty

28. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making*. 2006;26:565-574. DOI: `10.1177/0272989X06295361`.
29. Van Calster B, Vickers AJ. Calibration of risk prediction models: impact on decision-analytic performance. *Med Decis Making*. 2015;35:162-169. DOI: `10.1177/0272989X14547233`.
30. Efron B. Bootstrap methods: another look at the jackknife. *Ann Stat*. 1979;7:1-26. DOI: `10.1214/aos/1176344552`.
31. Riley RD, Debray TPA, Collins GS, Archer L, Ensor J, van Smeden M, Snell KIE. Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Stat Med*. 2021;40:4230-4251. DOI: `10.1002/sim.9025`.
32. Snell KIE, Archer L, Ensor J, Bonnett LJ, Debray TPA, Phillips B, Collins GS, Riley RD. External validation of clinical prediction models: simulation-based sample size calculations were more reliable than rules-of-thumb. *J Clin Epidemiol*. 2021;135:79-89. DOI: `10.1016/j.jclinepi.2021.02.011`.

### Data resources and software

33. Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data*. 2023;10:1. DOI: `10.1038/s41597-022-01899-x`.
34. Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV. PhysioNet. 2024. Version 3.1. DOI: `10.13026/kpb9-mt58`.
35. Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Sci Data*. 2018;5:180178. DOI: `10.1038/sdata.2018.178`.
36. Pollard T, Johnson A, Raffa J, Celi LA, Badawi O, Mark R. eICU Collaborative Research Database. PhysioNet. 2019. Version 2.0. DOI: `10.13026/C2WM1R`.
37. National Center for Health Statistics. National Health and Nutrition Examination Survey. URL: `https://www.cdc.gov/nchs/nhanes/`.
38. National Center for Health Statistics. Public-use linked mortality files. URL: `https://www.cdc.gov/nchs/linked-data/mortality-files/index.html`.
39. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res*. 2011;12:2825-2830. DOI: none assigned by JMLR; ACM index DOI: `10.5555/1953048.2078195`.

### Diabetes burden and mortality context

40. Seshasai SRK, Kaptoge S, Thompson A, Di Angelantonio E, Gao P, Sarwar N, et al. Diabetes mellitus, fasting glucose, and risk of cause-specific death. *N Engl J Med*. 2011;364:829-841. DOI: `10.1056/NEJMoa1008862`.
41. GBD 2021 Diabetes Collaborators. Global, regional, and national burden of diabetes from 1990 to 2021, with projections of prevalence to 2050. *Lancet*. 2023;402:203-234. DOI: `10.1016/S0140-6736(23)01301-6`.
42. GBD 2019 Diabetes Mortality Collaborators. Diabetes mortality and trends before 25 years of age: an analysis of the Global Burden of Disease Study 2019. *Lancet Diabetes Endocrinol*. 2022;10:177-192. DOI: `10.1016/S2213-8587(21)00349-1`.

## Recommended handling of uncertain items

- Replace the unverified `van Smeden 2019 JCE external validation` item with Debray 2017 BMJ, Collins 2014 BMC Med Res Methodol, or Binuya 2022 BMC Med Res Methodol (`10.1186/s12874-022-01801-8`) depending on the sentence. For the Introduction first paragraph, Debray 2017 and Collins 2014 are the cleaner choices.
- Use Nestor 2019 as PMLR/MLHC without DOI; this is normal for PMLR proceedings.
- Use Finlayson 2021 NEJM for dataset shift; use Finlayson 2019 Science only if the sentence explicitly mentions adversarial or gaming risk.
- For GBD, the 2023 Lancet GBD 2021 paper is stronger for current global burden. The GBD 2019 mortality paper is narrower and optional.
