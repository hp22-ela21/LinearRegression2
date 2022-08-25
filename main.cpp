/**************************************************************************************************
* main.cpp: Implementerar en modell som bygger på linjär regression via ett objekt av klassen
*           lin_reg. Träningsdata läses in från en textfil. Efter träningen har
*           slutförts så genomförs prediktion av alla indata inom ett angivet intervall,
*           vilket skrivs ut i terminalen.
*
*           Kompilera koden och skapa en körbar fil döpt main med följande kommando:
*           $ g++ *.cpp -o main -Wall
*
*           Kör sedan programmet med följande kommando:
*           $ ./main
**************************************************************************************************/
#include "lin_reg.hpp"

/**************************************************************************************************
* main: Implementerar en maskininlärningsmodell som baseras på linjär regression, där träningsdata 
*       läses in från en fil döpt data.txt. Regressionsmodellen tränas under 1000 epoker med en 
*       lärhastighet på 1 %. Modellen testas sedan för indata inom intervallet [-10, 10] med
*       en stegringshastighet på 0.5. Indata samt motsvarande predikterad utdata skrivs ut i 
*       terminalen. Resultatet indikerar att modellen efter träning predikterar med en precision
*       på 100 %, vilket innbär att träningen var lyckad.
**************************************************************************************************/
int main(void)
{
   lin_reg l1(1000, 0.01);
   l1.load_training_data("data.txt");
   l1.train();
   l1.predict_range(-10, 10, 0.5);
   return 0;
}