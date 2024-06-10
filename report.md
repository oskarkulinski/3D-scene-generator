1. Wyniki Testowe i Treningowe: Wyniki najlepiej oglądnąć włączając model
 i podziwiając wygenerowane obrazy. Z grubsza przypominają budynki, 
ale brak szczegółów i kształty są nierealistyczne.   
     

2. Uzasadnienie Wyboru Techniki/Modelu: Wybraliśmy architekturę GAN, ze względu na
jej prostotę, łatwość kontrolowania efektów, oraz dużą możliwość dostosowywania parametrów.   

     
3. Strategia Podziału Danych: Nie dzielimy danych, testowania dokonujemy wizualnie, 
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
obiecujące niż na pokojach. Prostsze kształty i mniejsza różnorodność bardzo pozytywnie wpłynęły
na wyniki modelu


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
W obec tego jednocześnie spróbowaliśmy zmienić dataset na zdjęcia architektury. Na tym datasecie efekty były
w bardzo krótkim czasie całkiem zadowalające. Jako że na datasecie pokoi nie udało się od 5 dni zrobić
żadnego znaczącego postępu, podjęliśmy decyzję o przerzuceniu się na datest architektur i pozbyciu
się podziału na klasy, w celu uproszczenia modelu. Przetestowaliśmy jednocześnie wersję generatora
ze zwiększaniem rozmiaru zdjęć przy pomocy upsamplingu i strides=2 w transpose konwolucji.
Upsampling przynióśł niezłe rezultaty, ale po 200 epokach uczenie zakończyło się z powodu problemów
technicznych. Wersja 2 przyniosła bardzo dobre wyniki, generowała dosyć różnorodne zdjęcia, z których
wiele w sporym stopniu przypominało budynki. Po raz pierwszy ograniczeniem okazał się dyskriminator,
generator był w stanie regularnie go oszukiwać. Zadecydowaliśmy o dodaniu większej ilości filtrów w dyskriminatorze.

6. Podsumowanie uczenia:
- learning rate w okolicach x.e-4 dawał najlepsze efekty, przy większym nie było zbieżności, przy mniejszym 
uczył się za wolno
- w generatorze transpose convolution sprawdziło się najlepiej, w dyskriminatorze zwykła convolucja
- strides w konwolucji nieco stabilizował uczenie się, w porównaniu do upsamplingu/downsamplingu
- dropout przy pewnych modelach poprawiał rezultaty, ale gdy zmieniliśmy dataset doprowadził do
generowania jednokolorowych plam
- batch normalization poprawiało wyniki, z momentum 0.8, większe powodowowało zbyt wolne uczenie, mniejsze prowadziło
do niestabilnośći
- 1000 epok mało kiedy poprawiało wyniki, po około 500 poprawki były niewielkie
- warstwy dense w generatorze lub dyskriminatorze pogarszało wyniki