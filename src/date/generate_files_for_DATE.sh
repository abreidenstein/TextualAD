#!/bin/bash
# Generates outlier files that are needed to run DATE
# 20 Newsgroups
cat ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/rel.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/comp-outliers.txt
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/rel.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/misc-outliers.txt
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/rel.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/pol-outliers.txt
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rel.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/rec-outliers.txt
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/rel-outliers.txt
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/rel.txt > ../../data/20ng_od/test/sci-outliers.txt

# AG News
cat ../../data/ag_od/test/sci.txt ../../data/ag_od/test/sports.txt ../../data/ag_od/test/world.txt > ../../data/ag_od/test/business-outliers.txt
cat ../../data/ag_od/test/business.txt ../../data/ag_od/test/sports.txt ../../data/ag_od/test/world.txt > ../../data/ag_od/test/sci-outliers.txt
cat ../../data/ag_od/test/business.txt ../../data/ag_od/test/sci.txt ../../data/ag_od/test/world.txt > ../../data/ag_od/test/sports-outliers.txt
cat ../../data/ag_od/test/business.txt ../../data/ag_od/test/sci.txt ../../data/ag_od/test/sports.txt > ../../data/ag_od/test/world-outliers.txt

# RNCP
cat ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/1-environnement-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/2-defense-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/3-patrimoine-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/4-economie-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/5-recherche-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/6-nautisme-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/7-aeronautique-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/8-securite-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/9-multimedia-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/10-humanitaire-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/11-nucleaire-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/12-enfance-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/13-saisonnier-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/14-assistance-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/15-sport-outliers.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt > ../../data/rncp_od/test/16-ingenierie-outliers.txt


# Generates additional files with all the complete train and test sets -needed to save DATE representations of these files and to train a tokenizer with them (for RNCP)
cat ../../data/20ng_od/test/comp.txt ../../data/20ng_od/test/misc.txt ../../data/20ng_od/test/pol.txt ../../data/20ng_od/test/rec.txt ../../data/20ng_od/test/rel.txt ../../data/20ng_od/test/sci.txt > ../../data/20ng_od/test/20ng_fullTestSet.txt
cat ../../data/ag_od/test/business.txt ../../data/ag_od/test/sci.txt ../../data/ag_od/test/sports.txt ../../data/ag_od/test/world.txt > ../../data/ag_od/test/ag_fullTestSet.txt
cat ../../data/rncp_od/test/1-environnement.txt ../../data/rncp_od/test/2-defense.txt ../../data/rncp_od/test/3-patrimoine.txt ../../data/rncp_od/test/4-economie.txt ../../data/rncp_od/test/5-recherche.txt ../../data/rncp_od/test/6-nautisme.txt ../../data/rncp_od/test/7-aeronautique.txt ../../data/rncp_od/test/8-securite.txt ../../data/rncp_od/test/9-multimedia.txt ../../data/rncp_od/test/10-humanitaire.txt ../../data/rncp_od/test/11-nucleaire.txt ../../data/rncp_od/test/12-enfance.txt ../../data/rncp_od/test/13-saisonnier.txt ../../data/rncp_od/test/14-assistance.txt ../../data/rncp_od/test/15-sport.txt ../../data/rncp_od/test/16-ingenierie.txt > ../../data/rncp_od/test/rncp_fullTestSet.txt

cat ../../data/20ng_od/train/0/comp.txt ../../data/20ng_od/train/0/misc.txt ../../data/20ng_od/train/0/pol.txt ../../data/20ng_od/train/0/rec.txt ../../data/20ng_od/train/0/rel.txt ../../data/20ng_od/train/0/sci.txt > ../../data/20ng_od/train/20ng_fullTrainSet.txt
cat ../../data/ag_od/train/0/business.txt ../../data/ag_od/train/0/sci.txt ../../data/ag_od/train/0/sports.txt ../../data/ag_od/train/0/world.txt > ../../data/ag_od/train/ag_fullTrainSet.txt
cat ../../data/rncp_od/train/0/1-environnement.txt ../../data/rncp_od/train/0/2-defense.txt ../../data/rncp_od/train/0/3-patrimoine.txt ../../data/rncp_od/train/0/4-economie.txt ../../data/rncp_od/train/0/5-recherche.txt ../../data/rncp_od/train/0/6-nautisme.txt ../../data/rncp_od/train/0/7-aeronautique.txt ../../data/rncp_od/train/0/8-securite.txt ../../data/rncp_od/train/0/9-multimedia.txt ../../data/rncp_od/train/0/10-humanitaire.txt ../../data/rncp_od/train/0/11-nucleaire.txt ../../data/rncp_od/train/0/12-enfance.txt ../../data/rncp_od/train/0/13-saisonnier.txt ../../data/rncp_od/train/0/14-assistance.txt ../../data/rncp_od/train/0/15-sport.txt ../../data/rncp_od/train/0/16-ingenierie.txt > ../../data/rncp_od/train/rncp_fullTrainSet.txt