/**************************************************************************************************
* lin_reg.cpp: Inneh�ller medlemsfunktioner tillh�rande klassen lin_reg, vilket anv�nds f�r 
*              implementering av maskininl�rningsmodeller som baseras p� linj�r regression.
**************************************************************************************************/
#include "lin_reg.hpp"

/* Statiska funktioner: */
static bool char_is_digit(const char c);
static void retrieve_double(std::vector<double>& data, 
                            std::string& s);

/**************************************************************************************************
* extract: Extraherar tr�ningsdata i form av flyttal ur angiven str�ng och lagrar som tr�ningsdata 
*          f�r angiven regressionsmodell. Ifall tv� flyttal lyckas extraheras lagras dessa som en 
*          tr�ningsupps�ttning via vektorer m_train_in samt m_train_out. Index f�r varje lagrad
*          tr�ningsupps�ttning lagras ocks� via vektorn m_train_order f�r att enkelt kunna 
*          randomisera ordningsf�ljden f�r tr�ningsupps�ttningarna vid tr�ning utan att f�rflytta 
*          tr�ningsdatan, vilket genomf�rs f�r att minska risken att eventuella icke avsedda 
*          m�nster som f�rekommer i tr�ningsdatan skall p�verka tr�ningen av regressionsmodellen.
* 
*          - s: Str�ng inneh�llande de flyttal som skall extraheras.
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
* shuffle: Randomiserar den inb�rdes ordningsf�ljden f�r angiven regressionsmodells 
*          tr�ningsupps�ttningar genom att f�rflytta inneh�llet i vektorn m_train_order, som
*          lagrar index f�r respektive tr�ningsupps�ttning.
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
* optimize: Justerar parametrar f�r angiven regressionsmodell i syfte att minska aktuellt fel. 
*           Prediktion genomf�rs via given insignal, d�r predikterat v�rde j�mf�rs mot givet 
*           referensv�rde f�r att ber�kna aktuellt fel. Modellens parametrar justeras sedan med en 
*           br�kdel av felet, vilket avg�rs av l�rhastigheten. 
*
*           F�r vikten (k-v�rdet) tas aktuell insignal i �tanke g�llande graden av justering, 
*           d� viktens betydelse f�r aktuell fel st�r i direkt proportion med aktuell insignal 
*           (ju h�gre insignal, desto mer p�verkan har vikten p� predikterad utsignal och d�rmed 
*           eventuellt fel).
* 
*           - input    : Insignal fr�n tr�ningsdata som anv�nds f�r att genomf�ra prediktion.
*           - reference: Referensv�rde fr�n tr�ningsdatan, som anv�nds f�r att ber�kna aktuellt
*                        fel via j�mf�relse med predikterat v�rde.
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
* lin_reg: Konstruktor f�r klassen lin_reg, vilket anv�nds f�r att initiera
*                    en ny regressionsmodell som baseras p� linj�r regression. Angivet antal 
*                    epoker samt l�rhastighet lagras inf�r tr�ning. Tr�ningsdata m�ste dock 
*                    tillf�ras i efterhand via n�gon av medlemsfunktioner load_training_data
*                    (f�r att l�sa in tr�ningsupps�ttingarna fr�n en fil) eller set_training_data
*                    (f�r att passera tr�ningsdata via referenser till vektorer).
* 
*                    - num_epochs   : Antalet epoker/omg�ngar som skall genomf�ras vid tr�ning.
*                    - learning_rate: L�rhastighet, avg�r med hur stor andel av aktuellt fel som
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
* lin_reg: F�rflyttningskonstruktor, som medf�r f�rflyttning av minne fr�n en 
*                    regressionsmodell till en annan, i detta fall fr�n source till angivet 
*                    objekt this via anrop av funktionen std::move.
* 
*                    Inneh�llet lagrat av regressionsmodellen source kopieras till angiven modell 
*                    this, f�ljt av att source nollst�lls. Efter f�rflyttningen har d�rmed enbart 
*                    angiven modell this tillg�ng till minnet i fr�ga.
* 
*                    - source: Den regressionsmodell som minnet skall f�rflyttas fr�n.
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
* set_epochs: Uppdaterar antalet epoker som sker vid tr�ning f�r angiven regressionsmodell ifall 
*             angivet nytt antal �verstiger noll.
*
*             - num_epochs: Det nya antalet epoker som skall genomf�ras vid tr�ning.
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
* set_learning_rate: S�tter ny l�rhastighet f�r att justera parametrarna f�r angiven 
*                    regressionsmodell ifall angivet nytt v�rde �verstiger noll.
* 
*                    - learning_rate: Den nya l�rhastighet som skall anv�ndas f�r att justera
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
* load_training_data: L�ser in tr�ningsdata fr�n en fil via angiven fils�kv�g, extraherar denna
*                     data i form av flyttal och lagrar som tr�ningsupps�ttningar f�r angiven
*                     regressionsmodell.
*                    
*                     - filepath: Fils�kv�gen som tr�ningsdatan skall l�sas fr�n.
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
* set_training_data: Kopierar tr�ningsdata till angiven regressionsmodell fr�n refererade vektorer 
*                    samt lagrar index f�r respektive tr�ningsupps�ttning. Enbart fullst�ndiga 
*                    tr�ningsupps�ttningar d�r b�de in- och utsignal f�rekommer lagras.
* 
*                    - train_in : Inneh�ller insignaler f�r samtliga tr�ningsupps�ttningar.
*                    - train_out: Inneh�ller utsignaler f�r samtliga tr�ningsupps�ttningar.
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
* train: Tr�nar angiven regressionsmodell under angivet antal epoker. Inf�r varje ny epok 
*        randomiseras ordningsf�ljden p� tr�ningsupps�ttningarna f�r att undvika att eventuella 
*        m�nster som upptr�der i tr�ningsdatan skall p�verka tr�ningen av modellen. 
*
*        Varje varv optimeras modellens parametrar genom att en prediktion genomf�rs via en 
*        insignal fr�n tr�ningsdatan, d�r det predikterade v�rdet j�mf�rs med referensv�rdet 
*        fr�n tr�ningsdatan. Differensen mellan dessa v�rden utg�r aktuellt fel och parametrarna
*        justeras med en br�kdel av detta v�rde, beroende p� aktuell l�rhastighet.
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
* predict: Genomf�r prediktion med angiven regressionsmodell via angiven insignal och returnerar 
*          motsvarande predikterad utsignal i form av ett flytta.
* 
*          - input: Den insignal som prediktion skall genomf�ras p�.
**************************************************************************************************/
double lin_reg::predict(const double input)
{
   return m_weight * input + m_bias;
}

/**************************************************************************************************
* predict_all: Genomf�r prediktion med angiven regressionsmodell f�r samtliga insignaler som
*              f�rekommer i tr�ningsdatan och skriver ut motsvarande predikterade utsignaler via 
*              angiven utstr�m, d�r standardutenhet std::cout anv�nds som default f�r utskrift i
*              i terminalen. V�rden mycket n�ra noll [-threshold, threshold] avrundas till noll
*              f�r att undvika utskrift med ett flertal decimaler i on�dan, vilket annars sker 
*              just runt noll.
* 
*              - threshold: Tr�skelv�rde runt nollpunkten [-threshold, threshold], d�r
*                           predikterad v�rde skall avrundas till noll (default = 0.001).
*              - ostream  : Angiven utstr�m (default = std::cout).
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
* predict_range: Genomf�r prediktion med angiven regressionsmodell f�r insignaler inom intervallet 
*                mellan angivet start- och slutv�rde [start_val, end_val], d�r inkrementering av 
*                insignalen sker inom detta intervall med angivet stegv�rde step.
* 
*                Varje insignal samt motsvarande predikterat v�rde skrivs ut via angiven utstr�m,
*                d�r standardutenheten std::cout anv�nds som default f�r utskrift i terminalen. 
*                V�rden mycket n�ra noll [-threshold, threshold] avrundas till noll f�r att 
*                undvika utskrift med ett flertal decimaler i on�dan, vilket annars sker just 
*                runt noll.
* 
*                - start_val: Minv�rde f�r det intervall av insignaler som skall testas.
*                - end_val  : Maxv�rde f�r det intervall av insignaler som skall testas.
*                - step     : Stegv�rde/inkrementeringsv�rde f�r insignaler (default = 1.0).
*                - threshold: Tr�skelv�rde, d�r samtliga predikterade v�rden som ligger inom
*                             intervallet [-threshold, threshold] avrundas till noll f�r att 
*                             undvika utskrift med on�digt antal decimaler (default = 0.001).
*                - ostream:   Angiven utstr�m (default = std::cout).
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
* char_is_digit: Indikerar ifall givet tecken utg�r en siffra eller ett relaterat tecken, s�som
*                ett minustecken eller en punkt. Eftersom flyttal ibland matas in b�de med
*                punkt samt kommatecken s� utg�r b�da giltiga tecken.
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
* retrieve_double: Typomvandlar inneh�ll lagrat som text till ett flyttal och lagrar detta i
*                  angiven vektor. Innan typomvandlingen �ger rum ers�tts eventuella kommatecken 
*                  med punkt, vilket m�jligg�r att flyttal kan l�sas in b�de med punkt eller
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

