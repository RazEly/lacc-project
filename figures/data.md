--- measure FPRT (FPRT) ---
  wrote figures/titrated_german-gpt2_delta_ll_FPRT.png
measure   group  n_readers       rho        p     ci_lo    ci_hi
   FPRT biology         43  0.089790 0.566914 -0.232266 0.407017
   FPRT physics         32 -0.157373 0.389682 -0.485180 0.182066
  wrote figures/titrated_german-gpt2_kstar_FPRT.png
  [tercile] pop_k=256 aligned_k=256
measure     map         comparison  n_readers  mean_dll         z            p winner        p_adj  pop_index  aligned_index
   FPRT tercile titrated vs binary         75 -1.121833 -5.816681 6.002744e-09 binary 9.004116e-09          4              4
   FPRT tercile    titrated vs pop         75  0.000000  0.000000 1.000000e+00    pop 1.000000e+00          4              4
   FPRT tercile      binary vs pop         75  1.121833  5.816681 6.002744e-09 binary 9.004116e-09          4              4
  [isotonic] pop_k=256 aligned_k=256
measure      map         comparison  n_readers  mean_dll         z            p winner        p_adj  pop_index  aligned_index
   FPRT isotonic titrated vs binary         75 -1.149671 -5.931478 3.002196e-09 binary 9.004116e-09          4              4
   FPRT isotonic    titrated vs pop         75 -0.027838 -1.064006 2.873259e-01    pop 2.873259e-01          4              4
   FPRT isotonic      binary vs pop         75  1.121833  5.816681 6.002744e-09 binary 9.004116e-09          4              4

--- measure GP (RPD_inc) ---
  wrote figures/titrated_german-gpt2_delta_ll_GP.png
measure   group  n_readers      rho        p     ci_lo    ci_hi
     GP biology         43 0.212041 0.172236 -0.068154 0.483918
     GP physics         32 0.238482 0.188697 -0.121316 0.561185
  wrote figures/titrated_german-gpt2_kstar_GP.png
  [tercile] pop_k=256 aligned_k=256
measure     map         comparison  n_readers  mean_dll          z            p winner        p_adj  pop_index  aligned_index
     GP tercile titrated vs binary         75 -2.917164 -10.663773 1.503702e-26 binary 2.255553e-26          4              4
     GP tercile    titrated vs pop         75  0.000000   0.000000 1.000000e+00    pop 1.000000e+00          4              4
     GP tercile      binary vs pop         75  2.917164  10.663773 1.503702e-26 binary 2.255553e-26          4              4
  [isotonic] pop_k=256 aligned_k=256
measure      map         comparison  n_readers  mean_dll          z            p winner        p_adj  pop_index  aligned_index
     GP isotonic titrated vs binary         75 -2.957952 -10.222667 1.569617e-24 binary 2.354425e-24          4              4
     GP isotonic    titrated vs pop         75 -0.040788  -0.789803 4.296428e-01    pop 4.296428e-01          4              4
     GP isotonic      binary vs pop         75  2.917164  10.663773 1.503702e-26 binary 4.511107e-26          4              4

--- measure TFT (TFT) ---
  wrote figures/titrated_german-gpt2_delta_ll_TFT.png
measure   group  n_readers       rho        p     ci_lo    ci_hi
    TFT biology         43  0.215053 0.166080 -0.075713 0.495924
    TFT physics         32 -0.064454 0.725987 -0.419229 0.307643
  wrote figures/titrated_german-gpt2_kstar_TFT.png
  [tercile] pop_k=1024 aligned_k=256
measure     map         comparison  n_readers  mean_dll          z            p   winner        p_adj  pop_index  aligned_index
    TFT tercile titrated vs binary         75 -3.418781 -13.086450 3.935784e-39   binary 5.903676e-39          5              4
    TFT tercile    titrated vs pop         75  0.226695   2.895875 3.781028e-03 titrated 3.781028e-03          5              4
    TFT tercile      binary vs pop         75  3.645476  14.085581 4.658132e-45   binary 1.397440e-44          5              4
  [isotonic] pop_k=1024 aligned_k=256
measure      map         comparison  n_readers  mean_dll          z            p   winner        p_adj  pop_index  aligned_index
    TFT isotonic titrated vs binary         75 -3.537335 -13.856375 1.164055e-43   binary 1.746083e-43          5              4
    TFT isotonic    titrated vs pop         75  0.108141   1.627296 1.036742e-01 titrated 1.036742e-01          5              4
    TFT isotonic      binary vs pop         75  3.645476  14.085581 4.658132e-45   binary 1.397440e-44          5              4
