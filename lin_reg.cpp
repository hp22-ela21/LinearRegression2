/**************************************************************************************************
* lin_reg.cpp: Innehåller medlemsfunktioner tillhörande klassen lin_reg, vilket används för 
*              implementering av maskininlärningsmodeller som baseras på linjär regression.
**************************************************************************************************/
#include "lin_reg.hpp"

/* Statiska funktioner: */
static bool char_is_digit(const char c);
static void retrieve_double(std::vector<double>& data, 
                            std::string& s);

/**************************************************************************************************
* extract: Extraherar träningsdata i form av flyttal ur angiven sträng och lagrar som träningsdata 
*          för angiven regressionsmodell. Ifall två flyttal lyckas extraheras lagras dessa som en 
*          träningsuppsättning via vektorer m_train_in samt m_train_out. Index för varje lagrad
*          träningsuppsättning lagras också via vektorn m_train_order för att enkelt kunna 
*          randomisera ordningsföljden för träningsuppsättningarna vid träning utan att förflytta 
*          träningsdatan, vilket genomförs för att minska risken att eventuella icke avsedda 
*          mönster som förekommer i träningsdatan skall påverka träningen av regressionsmodellen.
* 
*          - s: Sträng innehållande de flyttal som skall extraheras.
**************************************************************************************************/
void lin_reg::extract(const std::string& s)
{
   std::string num_str;
   std::vector<double> data;

   for (auto& i : s)
   {
      if (char_is_digit(i))
      {
         num_str += i;
      }
      else
      {
         retrieve_double(data, num_str);
      }
   }

   retrieve_double(data, num_str);

   if (data.size() == 2)
   {
      m_train_in.push_back(data[0]);
      m_train_out.push_back(data[1]);
      m_train_order.push_back(m_train_order.size());
   }
   return;
}

/**************************************************************************************************
* shuffle: Randomiserar den inbördes ordningsföljden för angiven regressionsmodells 
*          träningsuppsättningar genom att förflytta innehållet i vektorn m_train_order, som
*          lagrar index för respektive träningsuppsättning.
**************************************************************************************************/
void lin_reg::shuffle(void)
{
   for (std::size_t i = 0; i < m_train_order.size(); ++i)
   {
      const auto r = rand() % m_train_order.size();
      const auto temp = m_train_order[i];
      m_train_order[i] = m_train_order[r];
      m_train_order[r] = temp;
   }
   return;
}

/**************************************************************************************************
* optimize: Justerar parametrar för angiven regressionsmodell i syfte att minska aktuellt fel. 
*           Prediktion genomförs via given insignal, där predikterat värde jämförs mot givet 
*           referensvärde för att beräkna aktuellt fel. Modellens parametrar justeras sedan med en 
*           bråkdel av felet, vilket avgörs av lärhastigheten. 
*
*           För vikten (k-värdet) tas aktuell insignal i åtanke gällande graden av justering, 
*           då viktens betydelse för aktuell fel står i direkt proportion med aktuell insignal 
*           (ju högre insignal, desto mer påverkan har vikten på predikterad utsignal och därmed 
*           eventuellt fel).
* 
*           - input    : Insignal från träningsdata som används för att genomföra prediktion.
*           - reference: Referensvärde från träningsdatan, som används för att beräkna aktuellt
*                        fel via jämförelse med predikterat värde.
**************************************************************************************************/
void lin_reg::optimize(const double input, 
                       const double reference)
{
   const auto prediction = m_weight * input + m_bias;
   const auto error = reference - prediction;
   const auto change_rate = error * m_learning_rate;
   m_bias += change_rate;
   m_weight += change_rate * input;
   return;
}

/**************************************************************************************************
* lin_reg: Konstruktor för klassen lin_reg, vilket används för att initiera
*                    en ny regressionsmodell som baseras på linjär regression. Angivet antal 
*                    epoker samt lärhastighet lagras inför träning. Träningsdata måste dock 
*                    tillföras i efterhand via någon av medlemsfunktioner load_training_data
*                    (för att läsa in träningsuppsättingarna från en fil) eller set_training_data
*                    (för att passera träningsdata via referenser till vektorer).
* 
*                    - num_epochs   : Antalet epoker/omgångar som skall genomföras vid träning.
*                    - learning_rate: Lärhastighet, avgör med hur stor andel av aktuellt fel som
*                                     modellens parametrar (bias och vikt) skall justeras.
**************************************************************************************************/
lin_reg::lin_reg(const std::size_t num_epochs, 
                 const double learning_rate)
{
   set_epochs(num_epochs);
   set_learning_rate(learning_rate);
   return;
}

/**************************************************************************************************
* lin_reg: Förflyttningskonstruktor, som medför förflyttning av minne från en 
*                    regressionsmodell till en annan, i detta fall från source till angivet 
*                    objekt this via anrop av funktionen std::move.
* 
*                    Innehållet lagrat av regressionsmodellen source kopieras till angiven modell 
*                    this, följt av att source nollställs. Efter förflyttningen har därmed enbart 
*                    angiven modell this tillgång till minnet i fråga.
* 
*                    - source: Den regressionsmodell som minnet skall förflyttas från.
**************************************************************************************************/
lin_reg::lin_reg(lin_reg&& source) noexcept
{
   this->m_train_in = source.m_train_in;
   this->m_train_out = source.m_train_out;
   this->m_train_order = source.m_train_order;
   this->m_weight = source.m_weight;
   this->m_bias = source.m_bias;
   this->m_learning_rate = source.m_learning_rate;
   this->m_num_epochs = source.m_num_epochs;  

   source.m_train_in.clear();
   source.m_train_out.clear();
   source.m_train_order.clear();
   source.m_weight = 0;
   source.m_bias = 0;
   source.m_learning_rate = 0;
   source.m_num_epochs = 0;
   return;
}

/**************************************************************************************************
* set_epochs: Uppdaterar antalet epoker som sker vid träning för angiven regressionsmodell ifall 
*             angivet nytt antal överstiger noll.
*
*             - num_epochs: Det nya antalet epoker som skall genomföras vid träning.
**************************************************************************************************/
void lin_reg::set_epochs(const std::size_t num_epochs)
{
   if (num_epochs > 0)
   {
      m_num_epochs = num_epochs;
   }
   return;
}

/**************************************************************************************************
* set_learning_rate: Sätter ny lärhastighet för att justera parametrarna för angiven 
*                    regressionsmodell ifall angivet nytt värde överstiger noll.
* 
*                    - learning_rate: Den nya lärhastighet som skall användas för att justera
*                                     modellens parametrar (bias och vikt) vid fel.
**************************************************************************************************/
void lin_reg::set_learning_rate(const double learning_rate)
{
   if (learning_rate > 0)
   {
      m_learning_rate = learning_rate;
   }
   return;
}

/**************************************************************************************************
* load_training_data: Läser in träningsdata från en fil via angiven filsökväg, extraherar denna
*                     data i form av flyttal och lagrar som träningsuppsättningar för angiven
*                     regressionsmodell.
*                    
*                     - filepath: Filsökvägen som träningsdatan skall läsas från.
**************************************************************************************************/
void lin_reg::load_training_data(const std::string& filepath)
{
   std::ifstream fstream(filepath, std::ios::in);

   if (!fstream)
   {
      std::cerr << "Could not open file at path " << filepath << "!\n\n";
   }
   else
   {
      std::string s;
      while (std::getline(fstream, s))
      {
         extract(s);
      }
   }
   return;
}

/**************************************************************************************************
* set_training_data: Kopierar träningsdata till angiven regressionsmodell från refererade vektorer 
*                    samt lagrar index för respektive träningsuppsättning. Enbart fullständiga 
*                    träningsuppsättningar där både in- och utsignal förekommer lagras.
* 
*                    - train_in : Innehåller insignaler för samtliga träningsuppsättningar.
*                    - train_out: Innehåller utsignaler för samtliga träningsuppsättningar.
**************************************************************************************************/
void lin_reg::set_training_data(const std::vector<double>& train_in,
                                const std::vector<double>& train_out)
{
   auto num_sets = train_in.size();

   if (train_in.size() > train_out.size())
   {
      num_sets = train_out.size();
   }

   m_train_in.resize(num_sets);
   m_train_out.resize(num_sets);
   m_train_order.resize(num_sets);

   for (std::size_t i = 0; i < num_sets; ++i)
   {
      m_train_in[i] = train_in[i];
      m_train_out[i] = train_out[i];
      m_train_order[i] = i;
   }
   return;
}

/**************************************************************************************************
* train: Tränar angiven regressionsmodell under angivet antal epoker. Inför varje ny epok 
*        randomiseras ordningsföljden på träningsuppsättningarna för att undvika att eventuella 
*        mönster som uppträder i träningsdatan skall påverka träningen av modellen. 
*
*        Varje varv optimeras modellens parametrar genom att en prediktion genomförs via en 
*        insignal från träningsdatan, där det predikterade värdet jämförs med referensvärdet 
*        från träningsdatan. Differensen mellan dessa värden utgör aktuellt fel och parametrarna
*        justeras med en bråkdel av detta värde, beroende på aktuell lärhastighet.
**************************************************************************************************/
void lin_reg::train(void)
{
   for (std::size_t i = 0; i < m_num_epochs; ++i)
   {
      shuffle();

      for (auto& j : m_train_order)
      {
         optimize(m_train_in[j], m_train_out[j]);
      }
   }
   return;
}

/**************************************************************************************************
* predict: Genomför prediktion med angiven regressionsmodell via angiven insignal och returnerar 
*          motsvarande predikterad utsignal i form av ett flytta.
* 
*          - input: Den insignal som prediktion skall genomföras på.
**************************************************************************************************/
double lin_reg::predict(const double input)
{
   return m_weight * input + m_bias;
}

/**************************************************************************************************
* predict_all: Genomför prediktion med angiven regressionsmodell för samtliga insignaler som
*              förekommer i träningsdatan och skriver ut motsvarande predikterade utsignaler via 
*              angiven utström, där standardutenhet std::cout används som default för utskrift i
*              i terminalen. Värden mycket nära noll [-threshold, threshold] avrundas till noll
*              för att undvika utskrift med ett flertal decimaler i onödan, vilket annars sker 
*              just runt noll.
* 
*              - threshold: Tröskelvärde runt nollpunkten [-threshold, threshold], där
*                           predikterad värde skall avrundas till noll (default = 0.001).
*              - ostream  : Angiven utström (default = std::cout).
**************************************************************************************************/
void lin_reg::predict_all(const double threshold, 
                          std::ostream& ostream)
{
   const auto* last = &m_train_in[m_train_in.size() - 1];
   ostream << "----------------------------------------------------------------------------\n";

   for (auto& i : m_train_in)
   {
      const auto prediction = predict(i);
      ostream << "Input: " << i << "\n";

      if (prediction > -threshold && prediction < threshold)
      {
         ostream << "Output: " << 0 << "\n";
      }
      else
      {
         ostream << "Output: " << predict(i) << "\n";
      }

      if (&i < last) ostream << "\n";
   }

   ostream << "----------------------------------------------------------------------------\n\n";
   return;
}

/**************************************************************************************************
* predict_range: Genomför prediktion med angiven regressionsmodell för insignaler inom intervallet 
*                mellan angivet start- och slutvärde [start_val, end_val], där inkrementering av 
*                insignalen sker inom detta intervall med angivet stegvärde step.
* 
*                Varje insignal samt motsvarande predikterat värde skrivs ut via angiven utström,
*                där standardutenheten std::cout används som default för utskrift i terminalen. 
*                Värden mycket nära noll [-threshold, threshold] avrundas till noll för att 
*                undvika utskrift med ett flertal decimaler i onödan, vilket annars sker just 
*                runt noll.
* 
*                - start_val: Minvärde för det intervall av insignaler som skall testas.
*                - end_val  : Maxvärde för det intervall av insignaler som skall testas.
*                - step     : Stegvärde/inkrementeringsvärde för insignaler (default = 1.0).
*                - threshold: Tröskelvärde, där samtliga predikterade värden som ligger inom
*                             intervallet [-threshold, threshold] avrundas till noll för att 
*                             undvika utskrift med onödigt antal decimaler (default = 0.001).
*                - ostream:   Angiven utström (default = std::cout).
**************************************************************************************************/
void lin_reg::predict_range(const double start_val, 
                            const double end_val,
                            const double step, 
                            const double threshold, 
                            std::ostream& ostream)
{
   ostream << "----------------------------------------------------------------------------\n";

   for (double i = start_val; i <= end_val; i += step)
   {
      const auto prediction = predict(i);
      ostream << "Input: " << i << "\n";

      if (prediction > -threshold && prediction < threshold)
      {
         ostream << "Output: " << 0 << "\n";
      }
      else
      {
         ostream << "Output: " << predict(i) << "\n";
      } 

      if (i < end_val) ostream << "\n";
   }

   ostream << "----------------------------------------------------------------------------\n\n";
   return;
}

/**************************************************************************************************
* char_is_digit: Indikerar ifall givet tecken utgör en siffra eller ett relaterat tecken, såsom
*                ett minustecken eller en punkt. Eftersom flyttal ibland matas in både med
*                punkt samt kommatecken så utgör båda giltiga tecken.
* 
*                - c: Det tecken som skall kontrolleras.
**************************************************************************************************/
static bool char_is_digit(const char c)
{
   const auto s = "0123456789-.,";
   for (auto i = s; *i; ++i)
   {
      if (c == *i)
      {
         return true;
      }
   }
   return false;
}

/**************************************************************************************************
* retrieve_double: Typomvandlar innehåll lagrat som text till ett flyttal och lagrar detta i
*                  angiven vektor. Innan typomvandlingen äger rum ersätts eventuella kommatecken 
*                  med punkt, vilket möjliggör att flyttal kan läsas in både med punkt eller
*                  kommatecken som decimaltecken.
* 
*                  - data: Den vektor som typomvandlat flyttal skall lagras i.
*                  - s   : Data i form av text som skall typomvandlas till ett flyttal.
**************************************************************************************************/
static void retrieve_double(std::vector<double>& data, 
                            std::string& s)
{
   for (auto& i : s)
   {
      if (i == ',') i = '.';
   }

   try
   {
      const auto number = std::stod(s);
      data.push_back(number);
   }
   catch (std::invalid_argument&)
   {
      std::cerr << "Failed to convert " << s << " to double\n";
   }

   s.clear();
   return;
}

