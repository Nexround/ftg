## 模型信息
qwen2.5-0.5B 总参数494032768，24层tranformers块，隐藏层4864，共计4864*24=116736个神经元
从triviaQA中获取19517个高贡献神经元，固定他们，再从剩下97219个神经元中随机选取10%的神经元进行微调，即微调9721个神经元。微调神经元占总参数量0.00196%，占总神经元数量8.32%。
## 数据样例
[ { "from": "system", "value": "Given a query and a piece of text, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related." }, <br>
{ "from": "human", "value": "`<<<Query>>>`\nQué término coloquial se utiliza para los residentes de plymouth?\n\n`<<<Context>>>`\nPeople from Plymouth are known as Plymothians or less formally as Janners. Its meaning is described as a person from Devon, deriving from Cousin Jan (the Devon form of John), but more particularly in naval circles anyone from the Plymouth area." },<br> { "from": "gpt", "value": "7" } ]
## 跟lightblue/lb-reranker-0.5B-v1.0在文本检索基准BEIR上的比较

| 数据集               | MAP@1                                  | MAP@3                                  | MAP@5                                  | MAP@10                                 | MAP@100 (%)                            |
| -------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| **arguana**          | <span style="color:green">+3.08</span> | <span style="color:green">+4.85</span> | <span style="color:green">+3.35</span> | <span style="color:green">+3.29</span> | <span style="color:green">+2.89</span> |
| **dbpedia-entity**   | <span style="color:green">+5.12</span> | <span style="color:green">+1.95</span> | <span style="color:green">+3.07</span> | <span style="color:green">+2.45</span> | <span style="color:green">+2.05</span> |
| **fiqa**             | <span style="color:green">+1.38</span> | <span style="color:green">+5.58</span> | <span style="color:green">+4.20</span> | <span style="color:green">+4.07</span> | <span style="color:green">+3.38</span> |
| **nfcorpus**         | <span style="color:green">+1.56</span> | <span style="color:green">+0.33</span> | <span style="color:green">+1.08</span> | <span style="color:green">+0.97</span> | <span style="color:green">+0.64</span> |
| **scidocs**          | <span style="color:red">-2.21</span>   | <span style="color:green">+5.98</span> | <span style="color:green">+0.25</span> | <span style="color:green">+0.28</span> | <span style="color:green">+0.18</span> |
| **scifact**          | <span style="color:red">-1.30</span>   | <span style="color:green">+0.86</span> | <span style="color:green">+0.60</span> | <span style="color:red">-0.03</span>   | <span style="color:red">-0.07</span>   |
| **trec-covid-v2**    | <span style="color:red">-0.71</span>   | <span style="color:green">+1.74</span> | <span style="color:green">+2.79</span> | <span style="color:green">+1.58</span> | <span style="color:green">+0.67</span> |
| **vihealthqa**       | <span style="color:green">+6.03</span> | <span style="color:green">+7.91</span> | <span style="color:green">+6.62</span> | <span style="color:green">+5.66</span> | <span style="color:green">+5.52</span> |
| **webis-touche2020** | <span style="color:green">+2.82</span> | <span style="color:red">-7.41</span>   | <span style="color:red">-4.03</span>   | <span style="color:green">+6.05</span> | <span style="color:green">+2.08</span> |

## lora r1a2
| 数据集           | MAP@1                                    | MAP@3                                    | MAP@5                                    | MAP@10                                   | MAP@100 (%)                             |
| ---------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | --------------------------------------- |
| arguana          | <span style="color:green">+3.08%</span>  | <span style="color:green">+9.58%</span>  | <span style="color:green">+5.89%</span>  | <span style="color:green">+5.61%</span>  | <span style="color:green">+5.14%</span> |
| dbpedia-entity   | <span style="color:green">+16.97%</span> | <span style="color:green">+8.56%</span>  | <span style="color:green">+6.14%</span>  | <span style="color:green">+6.59%</span>  | <span style="color:green">+5.37%</span> |
| fiqa             | <span style="color:green">+13.57%</span> | <span style="color:green">+11.59%</span> | <span style="color:green">+10.25%</span> | <span style="color:green">+10.28%</span> | <span style="color:green">+9.53%</span> |
| nfcorpus         | <span style="color:green">+1.05%</span>  | <span style="color:green">+2.15%</span>  | <span style="color:green">+1.98%</span>  | <span style="color:green">+1.56%</span>  | <span style="color:green">+0.60%</span> |
| scidocs          | <span style="color:green">+18.79%</span> | <span style="color:green">+16.80%</span> | <span style="color:green">+5.80%</span>  | <span style="color:green">+7.03%</span>  | <span style="color:green">+5.74%</span> |
| scifact          | <span style="color:red">-5.61%</span>    | <span style="color:red">-0.87%</span>    | <span style="color:red">-0.98%</span>    | <span style="color:red">-1.81%</span>    | <span style="color:red">-1.70%</span>   |
| trec-covid-v2    | <span style="color:green">+1.45%</span>  | <span style="color:green">+0.37%</span>  | <span style="color:green">+4.57%</span>  | <span style="color:green">+3.87%</span>  | <span style="color:green">+1.18%</span> |
| vihealthqa       | <span style="color:green">+1.01%</span>  | <span style="color:green">+1.83%</span>  | <span style="color:green">+0.89%</span>  | <span style="color:green">+0.93%</span>  | <span style="color:green">+1.07%</span> |
| webis-touche2020 | <span style="color:green">+17.15%</span> | <span style="color:green">+2.50%</span>  | <span style="color:green">+0.98%</span>  | <span style="color:green">+3.82%</span>  | <span style="color:green">+1.72%</span> |

## lora r8a16
| Dataset          | MAP@1                                    | MAP@3                                    | MAP@5                                    | MAP@10                                   | MAP@100                                  |
| ---------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| arguana          | <span style="color:red">-22.99%</span>   | <span style="color:red">-12.68%</span>   | <span style="color:red">-10.89%</span>   | <span style="color:red">-9.87%</span>    | <span style="color:red">-9.85%</span>    |
| dbpedia-entity   | <span style="color:green">+8.26%</span>  | <span style="color:green">+6.15%</span>  | <span style="color:green">+6.55%</span>  | <span style="color:green">+7.82%</span>  | <span style="color:green">+7.47%</span>  |
| fiqa             | <span style="color:green">+11.63%</span> | <span style="color:green">+9.74%</span>  | <span style="color:green">+7.89%</span>  | <span style="color:green">+6.92%</span>  | <span style="color:green">+6.53%</span>  |
| nfcorpus         | <span style="color:green">+0.84%</span>  | <span style="color:red">-1.98%</span>    | <span style="color:red">-0.26%</span>    | <span style="color:red">-0.33%</span>    | <span style="color:red">-0.35%</span>    |
| scidocs          | <span style="color:grey">0.00%</span>    | <span style="color:green">+3.18%</span>  | <span style="color:red">-1.83%</span>    | <span style="color:green">+2.08%</span>  | <span style="color:green">+0.75%</span>  |
| scifact          | <span style="color:red">-3.19%</span>    | <span style="color:red">-0.72%</span>    | <span style="color:red">-0.52%</span>    | <span style="color:red">-0.55%</span>    | <span style="color:red">-0.68%</span>    |
| trec-covid-v2    | <span style="color:red">-2.10%</span>    | <span style="color:green">+1.23%</span>  | <span style="color:green">+2.95%</span>  | <span style="color:green">+3.61%</span>  | <span style="color:green">+1.37%</span>  |
| vihealthqa       | <span style="color:green">+3.82%</span>  | <span style="color:green">+4.50%</span>  | <span style="color:green">+3.75%</span>  | <span style="color:green">+3.29%</span>  | <span style="color:green">+3.48%</span>  |
| webis-touche2020 | <span style="color:green">+31.25%</span> | <span style="color:green">+36.24%</span> | <span style="color:green">+38.64%</span> | <span style="color:green">+32.58%</span> | <span style="color:green">+17.00%</span> |

## Qwen2.5-0.5B-Instruct的评测结果
| dataset                                           | version | metric           | mode | Qwen2.5-0.5B-Instruct_hf |
| ------------------------------------------------- | ------- | ---------------- | ---- | ------------------------ |
| TheoremQA                                         | 6f0af8  | score            | gen  | 13.25                    |
| gsm8k                                             | 1d7fe4  | accuracy         | gen  | 39.65                    |
| triviaqa                                          | 2121ce  | score            | gen  | 24.37                    |
| lukaemon_mmlu_college_biology                     | caec7d  | accuracy         | gen  | 46.53                    |
| lukaemon_mmlu_college_chemistry                   | 520aa6  | accuracy         | gen  | 32.00                    |
| lukaemon_mmlu_college_computer_science            | 99c216  | accuracy         | gen  | 37.00                    |
| lukaemon_mmlu_college_mathematics                 | 678751  | accuracy         | gen  | 35.00                    |
| lukaemon_mmlu_college_physics                     | 4f382c  | accuracy         | gen  | 24.51                    |
| lukaemon_mmlu_electrical_engineering              | 770ce3  | accuracy         | gen  | 51.03                    |
| lukaemon_mmlu_astronomy                           | d3ee01  | accuracy         | gen  | 47.37                    |
| lukaemon_mmlu_anatomy                             | 72183b  | accuracy         | gen  | 45.19                    |
| lukaemon_mmlu_abstract_algebra                    | 2db373  | accuracy         | gen  | 30.00                    |
| lukaemon_mmlu_machine_learning                    | 0283bb  | accuracy         | gen  | 43.75                    |
| lukaemon_mmlu_clinical_knowledge                  | cb3218  | accuracy         | gen  | 53.58                    |
| lukaemon_mmlu_global_facts                        | ab07b6  | accuracy         | gen  | 35.00                    |
| lukaemon_mmlu_management                          | 80876d  | accuracy         | gen  | 60.19                    |
| lukaemon_mmlu_nutrition                           | 4543bd  | accuracy         | gen  | 51.63                    |
| lukaemon_mmlu_marketing                           | 7394e3  | accuracy         | gen  | 75.21                    |
| lukaemon_mmlu_professional_accounting             | 444b7f  | accuracy         | gen  | 36.17                    |
| lukaemon_mmlu_high_school_geography               | 0780e6  | accuracy         | gen  | 55.05                    |
| lukaemon_mmlu_international_law                   | cf3179  | accuracy         | gen  | 70.25                    |
| lukaemon_mmlu_moral_scenarios                     | f6dbe2  | accuracy         | gen  | 25.59                    |
| lukaemon_mmlu_computer_security                   | ce7550  | accuracy         | gen  | 67.00                    |
| lukaemon_mmlu_high_school_microeconomics          | 04d21a  | accuracy         | gen  | 40.76                    |
| lukaemon_mmlu_professional_law                    | 5f7e6c  | accuracy         | gen  | 36.18                    |
| lukaemon_mmlu_medical_genetics                    | 881ef5  | accuracy         | gen  | 49.00                    |
| lukaemon_mmlu_professional_psychology             | 221a16  | accuracy         | gen  | 44.61                    |
| lukaemon_mmlu_jurisprudence                       | 001f24  | accuracy         | gen  | 52.78                    |
| lukaemon_mmlu_world_religions                     | 232c09  | accuracy         | gen  | 54.97                    |
| lukaemon_mmlu_philosophy                          | 08042b  | accuracy         | gen  | 45.98                    |
| lukaemon_mmlu_virology                            | 12e270  | accuracy         | gen  | 42.77                    |
| lukaemon_mmlu_high_school_chemistry               | ae8820  | accuracy         | gen  | 36.95                    |
| lukaemon_mmlu_public_relations                    | e7d39b  | accuracy         | gen  | 52.73                    |
| lukaemon_mmlu_high_school_macroeconomics          | a01685  | accuracy         | gen  | 43.33                    |
| lukaemon_mmlu_human_sexuality                     | 42407c  | accuracy         | gen  | 51.15                    |
| lukaemon_mmlu_elementary_mathematics              | 269926  | accuracy         | gen  | 37.04                    |
| lukaemon_mmlu_high_school_physics                 | 93278f  | accuracy         | gen  | 27.15                    |
| lukaemon_mmlu_high_school_computer_science        | 9965a5  | accuracy         | gen  | 48.00                    |
| lukaemon_mmlu_high_school_european_history        | eefc90  | accuracy         | gen  | 59.39                    |
| lukaemon_mmlu_business_ethics                     | 1dec08  | accuracy         | gen  | 49.00                    |
| lukaemon_mmlu_moral_disputes                      | a2173e  | accuracy         | gen  | 48.84                    |
| lukaemon_mmlu_high_school_statistics              | 8f3f3a  | accuracy         | gen  | 30.09                    |
| lukaemon_mmlu_miscellaneous                       | 935647  | accuracy         | gen  | 54.66                    |
| lukaemon_mmlu_formal_logic                        | cfcb0c  | accuracy         | gen  | 33.33                    |
| lukaemon_mmlu_high_school_government_and_politics | 3c52f9  | accuracy         | gen  | 53.37                    |
| lukaemon_mmlu_prehistory                          | bbb197  | accuracy         | gen  | 52.78                    |
| lukaemon_mmlu_security_studies                    | 9b1743  | accuracy         | gen  | 47.76                    |
| lukaemon_mmlu_high_school_biology                 | 37b125  | accuracy         | gen  | 49.68                    |
| lukaemon_mmlu_logical_fallacies                   | 9cebb0  | accuracy         | gen  | 51.53                    |
| lukaemon_mmlu_high_school_world_history           | 048e7e  | accuracy         | gen  | 58.65                    |
| lukaemon_mmlu_professional_medicine               | 857144  | accuracy         | gen  | 36.40                    |
| lukaemon_mmlu_high_school_mathematics             | ed4dc0  | accuracy         | gen  | 32.59                    |
| lukaemon_mmlu_college_medicine                    | 38709e  | accuracy         | gen  | 43.35                    |
| lukaemon_mmlu_high_school_us_history              | 8932df  | accuracy         | gen  | 52.94                    |
| lukaemon_mmlu_sociology                           | c266a2  | accuracy         | gen  | 65.67                    |
| lukaemon_mmlu_econometrics                        | d1134d  | accuracy         | gen  | 28.95                    |
| lukaemon_mmlu_high_school_psychology              | 7db114  | accuracy         | gen  | 62.20                    |
| lukaemon_mmlu_human_aging                         | 82a410  | accuracy         | gen  | 46.19                    |
| lukaemon_mmlu_us_foreign_policy                   | 528cfe  | accuracy         | gen  | 73.00                    |
| lukaemon_mmlu_conceptual_physics                  | 63588e  | accuracy         | gen  | 39.57                    |
| truthful_qa                                       | 5ddc62  | bleu_max         | gen  | 0.01                     |
| truthful_qa                                       | 5ddc62  | bleu_diff        | gen  | -0.00                    |
| truthful_qa                                       | 5ddc62  | bleu_acc         | gen  | 0.15                     |
| hellaswag                                         | 6faab5  | accuracy         | gen  | 30.98                    |
| mmlu-humanities                                   | -       | naive_average    | gen  | 49.48                    |
| mmlu-stem                                         | -       | naive_average    | gen  | 40.02                    |
| mmlu-social-science                               | -       | naive_average    | gen  | 51.55                    |
| mmlu-other                                        | -       | naive_average    | gen  | 48.71                    |
| mmlu                                              | -       | naive_average    | gen  | 46.59                    |
| mmlu-weighted                                     | -       | weighted_average | gen  | 45.19                    |
## 基于我们方法微调模型的评测结果

| dataset                                           | version | metric           | mode | checkpoint-285430_hf |
| ------------------------------------------------- | ------- | ---------------- | ---- | -------------------- |
| TheoremQA                                         | 6f0af8  | score            | gen  | 11.38                |
| gsm8k                                             | 1d7fe4  | accuracy         | gen  | 27.75                |
| triviaqa                                          | 2121ce  | score            | gen  | 4.79                 |
| lukaemon_mmlu_college_biology                     | caec7d  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_college_chemistry                   | 520aa6  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_college_computer_science            | 99c216  | accuracy         | gen  | 1.00                 |
| lukaemon_mmlu_college_mathematics                 | 678751  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_college_physics                     | 4f382c  | accuracy         | gen  | 2.94                 |
| lukaemon_mmlu_electrical_engineering              | 770ce3  | accuracy         | gen  | 2.76                 |
| lukaemon_mmlu_astronomy                           | d3ee01  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_anatomy                             | 72183b  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_abstract_algebra                    | 2db373  | accuracy         | gen  | 3.00                 |
| lukaemon_mmlu_machine_learning                    | 0283bb  | accuracy         | gen  | 1.79                 |
| lukaemon_mmlu_clinical_knowledge                  | cb3218  | accuracy         | gen  | 0.38                 |
| lukaemon_mmlu_global_facts                        | ab07b6  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_management                          | 80876d  | accuracy         | gen  | 0.97                 |
| lukaemon_mmlu_nutrition                           | 4543bd  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_marketing                           | 7394e3  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_professional_accounting             | 444b7f  | accuracy         | gen  | 0.35                 |
| lukaemon_mmlu_high_school_geography               | 0780e6  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_international_law                   | cf3179  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_moral_scenarios                     | f6dbe2  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_computer_security                   | ce7550  | accuracy         | gen  | 1.00                 |
| lukaemon_mmlu_high_school_microeconomics          | 04d21a  | accuracy         | gen  | 0.84                 |
| lukaemon_mmlu_professional_law                    | 5f7e6c  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_medical_genetics                    | 881ef5  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_professional_psychology             | 221a16  | accuracy         | gen  | 0.33                 |
| lukaemon_mmlu_jurisprudence                       | 001f24  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_world_religions                     | 232c09  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_philosophy                          | 08042b  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_virology                            | 12e270  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_chemistry               | ae8820  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_public_relations                    | e7d39b  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_macroeconomics          | a01685  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_human_sexuality                     | 42407c  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_elementary_mathematics              | 269926  | accuracy         | gen  | 3.17                 |
| lukaemon_mmlu_high_school_physics                 | 93278f  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_computer_science        | 9965a5  | accuracy         | gen  | 1.00                 |
| lukaemon_mmlu_high_school_european_history        | eefc90  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_business_ethics                     | 1dec08  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_moral_disputes                      | a2173e  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_statistics              | 8f3f3a  | accuracy         | gen  | 1.85                 |
| lukaemon_mmlu_miscellaneous                       | 935647  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_formal_logic                        | cfcb0c  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_government_and_politics | 3c52f9  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_prehistory                          | bbb197  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_security_studies                    | 9b1743  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_biology                 | 37b125  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_logical_fallacies                   | 9cebb0  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_world_history           | 048e7e  | accuracy         | gen  | 0.42                 |
| lukaemon_mmlu_professional_medicine               | 857144  | accuracy         | gen  | 2.21                 |
| lukaemon_mmlu_high_school_mathematics             | ed4dc0  | accuracy         | gen  | 0.74                 |
| lukaemon_mmlu_college_medicine                    | 38709e  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_high_school_us_history              | 8932df  | accuracy         | gen  | 0.49                 |
| lukaemon_mmlu_sociology                           | c266a2  | accuracy         | gen  | 0.50                 |
| lukaemon_mmlu_econometrics                        | d1134d  | accuracy         | gen  | 0.88                 |
| lukaemon_mmlu_high_school_psychology              | 7db114  | accuracy         | gen  | 0.18                 |
| lukaemon_mmlu_human_aging                         | 82a410  | accuracy         | gen  | 1.79                 |
| lukaemon_mmlu_us_foreign_policy                   | 528cfe  | accuracy         | gen  | 0.00                 |
| lukaemon_mmlu_conceptual_physics                  | 63588e  | accuracy         | gen  | 0.00                 |
| truthful_qa                                       | 5ddc62  | bleu_max         | gen  | 0.01                 |
| truthful_qa                                       | 5ddc62  | bleu_diff        | gen  | -0.00                |
| truthful_qa                                       | 5ddc62  | bleu_acc         | gen  | 0.06                 |
| hellaswag                                         | 6faab5  | accuracy         | gen  | 4.75                 |
| mmlu-humanities                                   | -       | naive_average    | gen  | 0.07                 |
| mmlu-stem                                         | -       | naive_average    | gen  | 1.01                 |
| mmlu-social-science                               | -       | naive_average    | gen  | 0.23                 |
| mmlu-other                                        | -       | naive_average    | gen  | 0.44                 |
| mmlu                                              | -       | naive_average    | gen  | 0.50                 |
| mmlu-weighted                                     | -       | weighted_average | gen  | 0.39                 |

## lightblue/lb-reranker-0.5B-v1.0的评测结果

| dataset                                           | version | metric           | mode | lb-reranker-0.5B-v1.0_hf |
| ------------------------------------------------- | ------- | ---------------- | ---- | ------------------------ |
| TheoremQA                                         | 6f0af8  | score            | gen  | 0.38                     |
| gsm8k                                             | 1d7fe4  | accuracy         | gen  | 2.65                     |
| triviaqa                                          | 2121ce  | score            | gen  | 4.65                     |
| lukaemon_mmlu_college_biology                     | caec7d  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_college_chemistry                   | 520aa6  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_college_computer_science            | 99c216  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_college_mathematics                 | 678751  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_college_physics                     | 4f382c  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_electrical_engineering              | 770ce3  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_astronomy                           | d3ee01  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_anatomy                             | 72183b  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_abstract_algebra                    | 2db373  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_machine_learning                    | 0283bb  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_clinical_knowledge                  | cb3218  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_global_facts                        | ab07b6  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_management                          | 80876d  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_nutrition                           | 4543bd  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_marketing                           | 7394e3  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_professional_accounting             | 444b7f  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_geography               | 0780e6  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_international_law                   | cf3179  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_moral_scenarios                     | f6dbe2  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_computer_security                   | ce7550  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_microeconomics          | 04d21a  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_professional_law                    | 5f7e6c  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_medical_genetics                    | 881ef5  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_professional_psychology             | 221a16  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_jurisprudence                       | 001f24  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_world_religions                     | 232c09  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_philosophy                          | 08042b  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_virology                            | 12e270  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_chemistry               | ae8820  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_public_relations                    | e7d39b  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_macroeconomics          | a01685  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_human_sexuality                     | 42407c  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_elementary_mathematics              | 269926  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_physics                 | 93278f  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_computer_science        | 9965a5  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_european_history        | eefc90  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_business_ethics                     | 1dec08  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_moral_disputes                      | a2173e  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_statistics              | 8f3f3a  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_miscellaneous                       | 935647  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_formal_logic                        | cfcb0c  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_government_and_politics | 3c52f9  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_prehistory                          | bbb197  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_security_studies                    | 9b1743  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_biology                 | 37b125  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_logical_fallacies                   | 9cebb0  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_world_history           | 048e7e  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_professional_medicine               | 857144  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_mathematics             | ed4dc0  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_college_medicine                    | 38709e  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_us_history              | 8932df  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_sociology                           | c266a2  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_econometrics                        | d1134d  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_high_school_psychology              | 7db114  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_human_aging                         | 82a410  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_us_foreign_policy                   | 528cfe  | accuracy         | gen  | 0.00                     |
| lukaemon_mmlu_conceptual_physics                  | 63588e  | accuracy         | gen  | 0.00                     |
| truthful_qa                                       | 5ddc62  | bleu_max         | gen  | 0.00                     |
| truthful_qa                                       | 5ddc62  | bleu_diff        | gen  | 0.00                     |
| truthful_qa                                       | 5ddc62  | bleu_acc         | gen  | 0.00                     |
| hellaswag                                         | 6faab5  | accuracy         | gen  | 0.00                     |
| mmlu-humanities                                   | -       | naive_average    | gen  | 0.00                     |
| mmlu-stem                                         | -       | naive_average    | gen  | 0.00                     |
| mmlu-social-science                               | -       | naive_average    | gen  | 0.00                     |
| mmlu-other                                        | -       | naive_average    | gen  | 0.00                     |
| mmlu                                              | -       | naive_average    | gen  | 0.00                     |
| mmlu-weighted                                     | -       | weighted_average | gen  | 0.00                     |