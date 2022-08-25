/**************************************************************************************************
* lin_reg.hpp: Inneh�ller funktionalitet f�r implementering av maskininl�rningsmodeller som
*              baseras p� linj�r regression via klassen lin_reg. 
**************************************************************************************************/
#ifndef LIN_REG_HPP_
#define LIN_REG_HPP_

/* Inkluderingsdirektiv: */
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

/**************************************************************************************************
* lin_reg: Klass f�r implementering av maskininl�rningsmodeller som baseras p� linj�r regression. 
*          Tr�ningsdata med valfritt antal tr�ningsupps�ttningar kan l�sas in fr�n en fil eller 
*          passeras via referenser till vektorer.
* 
*          Klassens kopieringskonstruktor samt tilldelningsoperator �r raderade, vilket medf�r att 
*          minnet f�r ett givet objekt ej kan kopieras till/fr�n ett annat objekt. Klassens
*          f�rflyttningskonstruktor �r dock implementerad, vilket medf�r att minnet f�r ett 
*          givet objekt kan f�rflyttas till ett annat objekt via funktionen std::move. 
**************************************************************************************************/
class lin_reg
{
protected:
   /* Medlemmar: */
   std::vector<double> m_train_in;          /* Indata f�r tr�ningsupps�ttningar */
   std::vector<double> m_train_out;         /* Utdata f�r tr�ningsupps�ttningarna. */
   std::vector<std::size_t> m_train_order;  /* Ordningsf�ljd f�r tr�ningsupps�ttningarna. */
   double m_weight = 0;                     /* Lutning (k-v�rde). */
   double m_bias = 0;                       /* Vilov�rde (m-v�rde). */
   double m_learning_rate = 0;              /* L�rhastighet (avg�r justeringsgrad vid fel). */
   std::size_t m_num_epochs = 0;            /* Antalet tr�ningsomg�ngar. */

   /* Medlemsfunktioner: */
   void extract(const std::string& s);
   void shuffle(void);
   void optimize(const double input, 
                 const double output);
public:
   lin_reg(void) { }
   lin_reg(const std::size_t num_epochs, 
                     const double learning_rate);
   ~lin_reg(void) { }
   lin_reg(lin_reg&) = delete; 
   lin_reg& operator = (lin_reg&) = delete; 
   lin_reg(lin_reg&& source) noexcept;

   double weight(void) { return m_weight; }
   double bias(void) { return m_bias; }
   double learning_rate(void) { return m_learning_rate; }
   std::size_t epochs(void) { return m_num_epochs; }

   void set_epochs(const std::size_t num_epochs);
   void set_learning_rate(const double learning_rate);
   void load_training_data(const std::string& filepath);
   void set_training_data(const std::vector<double>& train_in, 
                          const std::vector<double>& train_out);
   void train(void);
   double predict(const double input);
   void predict_all(const double threshold = 0.001, 
                    std::ostream& ostream = std::cout);
   void predict_range(const double start_val, 
                      const double end_val, 
                      const double step = 1,
                      const double threshold = 0.001, 
                      std::ostream& ostream = std::cout);
};

#endif /* LIN_REG_HPP_ */