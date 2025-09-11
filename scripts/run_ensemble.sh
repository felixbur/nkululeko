#/bi/bash 
# ensemble based on performance
# based on ococosda 2024 paper
python3 -m nkululeko.ensemble \
examples/exp_kbes_audmodel.ini \
examples/exp_kbes_hubert.ini \
examples/exp_kbes_wavlm.ini \
--method performance_weighted \
--weights 0.77 0.775 0.758 