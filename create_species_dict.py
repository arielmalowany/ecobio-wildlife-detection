import json
import pandas as pd
import os 

os.chdir('/Users/arielmalowany/Desktop/Learning/Cupybara')

with open('/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/geofence_base.json', 'r') as j:
    data_dictionary = json.load(j)
    
search_species = 'sciuridae' 
    
keys_list = list(data_dictionary.keys())
found_key = [x for x in keys_list if x.find(search_species) > -1]
country_map = {
    "ARG": "Argentina",
    "BRA": "Brasil",
    "PRY": "Paraguay",
    "URY": "Uruguay",
    "CHL": "Chile"
}

standardized_species = pd.DataFrame(columns= ('Especie', 'Argentina', 'Brasil', 'Paraguay', 'Uruguay', 'Chile'))

for x in found_key:
 species_dict = data_dictionary.get(x)
 allowed_countries = species_dict.get("allow")
 standardized_species = pd.concat([standardized_species, pd.DataFrame([[x, 0, 0, 0, 0, 0]], columns= ('Especie', 'Argentina', 'Brasil', 'Paraguay', 'Uruguay', 'Chile'))], ignore_index= True)
 for code, name in country_map.items():
     if allowed_countries.get(code) is not None:
         standardized_species.loc[standardized_species['Especie'] == x, name] = 1

standardized_species.to_excel(f'{search_species}.xlsx')