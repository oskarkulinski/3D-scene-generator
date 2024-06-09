1. Wyniki Testowe i Treningowe: Wyniki najlepiej oglądnąć włączając model
 i podziwiając wygenerowane obrazy. Z grubsza przypominają budynki, 
ale brak szczegółów i kształty są nierealistyczne.   
     

2. Uzasadnienie Wyboru Techniki/Modelu: Wybraliśmy architekturę GAN, ze względu na
jej prostotę, łatwość kontrolowania efektów, oraz dużą możliwość dostosowywania parametrów.   

     
3. Strategia Podziału Danych: Nie dzielimy danych, testowania dokonujemy wzrokowo, 
ze względu na trudność zmierzenia efektów pracy GANa, bez osobnego modelu.
Nie używamy walidacji, gdyż nie mieliśmy na żadnym etapie pracy problemem z wynikami
dyskriminatora. Mieliśmy w planach dodanie walidacji, gdyby taki problem się pojawił,
ale nie zaszła taka potrzeba. W obec braku problemów z wynikami dyskriminatora, 
uznaliśmy za cenniejsze zwiększenie zestawu treningowego, niż walidację.


4. Opis Danych Wejściowych: Początkowo skorzystaliśmy ze zbioru obrazów pomieszczeń,
zawężonego do 5 klas, potem zwiększyliśmy ilość klas do 10, co znacząco pogorszyło wyniki,
obrazy były znacznie bardziej rozmazane i jedynie sporadycznie tworzyły się na nich
realistyczne obiekty, jak krzesła czy szafki z książkami.   
Ostatecznie zmieniliśmy dataset na zdjęcia budynków, gdyż nie udawało nam się osiągnąć
znacznego postępu przez 5 dni. Zdecydowaliśmy że zdjęcia pokojów mogą zawierać zbyt dużo
różnorodnych obiektów, a po krótkich testach wyniki na architekturze były znacznie bardziej
obiecujące niż na pokojach.


5. Analiza Wyników: Początkowo wyniki były fatalne, zmiany modelu nic nie dawały, po 2 dniach
okazało się, że błąd leżał w funkcji treningowej. Po poprawieniu wyniki zaczęły być coraz lepsze,
zmniejszyliśmy learning rate, dodaliśmy normalizację i dropout. Potem zmienialiśmy ilość warstw
i filtrów, ale nie udało się uzyskać realistycznych kształtów. Pomogło zmienienie konwolucję na
transpose konwolucję w generatorze. To znacznie poprawiło realistyczność wyników. Po dostosowaniu 
momentum w normalizacji (0.7 było za małe, 0.9 za duże, model uczył się bardzo wolno), 
beta_1 w learning rate na 0.5, zgodnie z zaleceniami w pracy https://arxiv.org/abs/1511.06434
efekty były niezłe, ale model regularnie po około 300 - 500 epoce przestawał robić jakiekolwiek
postępy, a nie generował zbyt realistycznych obrazów. Spróbowaliśmy uprościć dyskriminator, co lekko
pomogło, ale wyniki wciąż nie były zbyt realistyczne. Próbowaliśmy zmienić wykorzystywane klasy, a potem
zwiększyliśmy ilość klas do 10, ale to przyniosło rezultaty odwrotne do oczekiwanych, model tworzył bardziej
niewyraźne kszałty, i rzadziej generował obiekty typu krzesła czy szafki.   
W obec tego jednocześnie próbowaliśmy.