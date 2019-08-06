#include "DfaParser.h"
#include "Synthesizer.h"
#include "PythonFormatter.h"
#include "ProductDfa.h"
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "usage: ./shield_synthesizer <num_choices> <output_file> [<Path to DFA file>, ...]\n";
        return -1;
    }
    int num_choices = std::atoi(argv[1]);
    std::string output_file = argv[2];
    
    std::vector<Dfa*> dfas;
    for (int i = 3; i < argc; i++) {
        std::string file_name = argv[i];
        dfas.push_back(DfaParser::parse_dfa_from_file(file_name));
    }
    
    Dfa* dfa;
    if (dfas.size() == 1) {
        dfa = dfas[0];
    } else {
        dfa = new ProductDfa(dfas);
        for (auto dfa : dfas) {
            delete dfa;
        }
    }
     
    std::cout << "Specification Automaton:\n" << dfa->to_string() << std::endl;

    std::cout << "Creating monitor..." << std::endl;
    ShieldMonitor* s_monitor = new ShieldMonitor(dfa);
    std::cout << "Shield monitor states: " << s_monitor->nodes_.size() << std::endl;
    std::cout << "Shield Monitor:\n" << s_monitor->to_string() << std::endl;
    
    Synthesizer* synthesizer = new Synthesizer(s_monitor, dfa->num_inputs_, num_choices);
    
    PythonFormatter* formatter = new PythonFormatter();
    synthesizer->to_output_format(formatter);
    std::ofstream out_file(output_file + formatter->get_extension());
    formatter->write_to_stream(out_file);
    
    return 0;
}
