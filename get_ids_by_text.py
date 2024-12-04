import json

def carica_test_ids(file_test):
    """Carica gli ID dal file test.txt in un insieme."""
    with open(file_test, 'r', encoding='utf-8') as f:
        test_ids = set(line.strip() for line in f if line.strip())
    return test_ids

def carica_annotations(file_annotations):
    """Carica le annotazioni dal file JSON."""
    with open(file_annotations, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations

def cerca_stringa_nelle_annotations(annotations, stringa):
    """Trova gli ID che contengono la stringa specificata nel campo 'text'."""
    matching_ids = set()
    for id_key, data in annotations.items():
        for annotazione in data.get('annotations', []):
            if stringa.lower() in annotazione.get('text', '').lower():
                matching_ids.add(id_key)
                break  # Evita di aggiungere lo stesso ID più volte
    return matching_ids

def main():
    # Specifica i percorsi dei file
    file_annotations = 'annotations_kitml.json'
    file_test = 'test.txt'
    
    # Carica i dati
    test_ids = carica_test_ids(file_test)
    annotations = carica_annotations(file_annotations)
    
    # Chiedi all'utente di inserire la stringa da cercare
    stringa_da_cercare = "waving"
    
    # Trova gli ID che contengono la stringa
    ids_con_stringa = cerca_stringa_nelle_annotations(annotations, stringa_da_cercare)
    
    # Interseca con gli ID presenti in test.txt
    ids_finali = ids_con_stringa.intersection(test_ids)
    
    if ids_finali:
        print("Gli ID che contengono la stringa '{}' e sono presenti in test.txt sono:".format(stringa_da_cercare))
        for id_val in sorted(ids_finali):
            print(id_val)
    else:
        print("Nessun ID soddisfa entrambe le condizioni.")

if __name__ == "__main__":
    main()
