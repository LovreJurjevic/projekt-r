# Bilježnice korištene u projektu

## Opis bilježnica

+ ```01_data_preparation.ipynb```: Analizira datoteku ```fastener_dataset/annotations/instances_default.json``` kako bi izvukla podatke o putevima slika i odgovarajućim kategorijama kojoj pripadaju. Funkcionalnost bilježnice odgovara modulu ```src/data.py``` koju koriste kasnije bilježnice
+ ```02_dataset_creation.ipynb```: Od podataka nastalih u prvoj bilježnici rade se 'datasetovi' potrebni za učenje modela i njegovo testiranje. Jako je bitno da se nikad ne mijenjaju 'random seedovi' kako se ne bi kompromirao skup za testiranje sa slikama na kojima se model učio. Odgovara modulu ```src/dataset.py```
+ ```03_baseline_model.ipynb```: Bilježnica u kojoj se isprobava funkcionira li učenje na pripremljenim podacima
+ ```04_model_improvement.ipynb```: Bilježnica osmišljena za definiranje novih modela te njihovo učenje, a naposljetku i spremanje modela na disk
+ ```05_testing.ipynb```: Proučavanje rezultata