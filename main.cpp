/**************************************************************************************************
* main.cpp: Implementerar en modell som bygger p� linj�r regression via ett objekt av klassen
*           lin_reg. Tr�ningsdata l�ses in fr�n en textfil. Efter tr�ningen har
*           slutf�rts s� genomf�rs prediktion av alla indata inom ett angivet intervall,
*           vilket skrivs ut i terminalen.
*
*           I Windows, kompilera koden och skapa en k�rbar fil main.exe med f�ljande kommando:
*           $ g++ main.cpp lin_reg.cpp -o main.exe -Wall
*
*           K�r sedan programmet med f�ljande kommando:
*           $ main.exe
**************************************************************************************************/
#include "lin_reg.hpp"

/**************************************************************************************************
* main: Implementerar en maskininl�rningsmodell som baseras p� linj�r regression, d�r tr�ningsdata 
*       l�ses in fr�n en fil d�pt data.txt. Regressionsmodellen tr�nas under 1000 epoker med en 
*       l�rhastighet p� 1 %. Modellen testas sedan f�r indata inom intervallet [-10, 10] med
*       en stegringshastighet p� 0.5. Indata samt motsvarande predikterad utdata skrivs ut i 
*       terminalen. Resultatet indikerar att modellen efter tr�ning predikterar med en precision
*       p� 100 %, vilket inneb�r att tr�ningen var lyckad.
**************************************************************************************************/
int main(void)
{
   lin_reg l1(1000, 0.01);
   l1.load_training_data("data.txt");
   l1.train();
   l1.predict_range(-10, 10, 0.5);
   return 0;
}