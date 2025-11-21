import onnxruntime
import numpy as np

# 1. Carica il modello ONNX (il tuo algoritmo)
session = onnxruntime.InferenceSession("Fratao_Logic_Model.onnx")

# Ottieni i nomi degli input e degli output del modello (necessari per ONNX Runtime)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Modello Fratao caricato. Input previsto: {input_name}")

# 2. Prepara i Dati di Test (Dimostrazione del Funzionamento)
# Testiamo 2 scenari diversi per dimostrare che il peso dei "Dati Conosciuti" è basso.

# Scenario A: Basso Dato Essenziale, Alto Dato Conosciuto (Dovrebbe dare un risultato Basso)
data_A = np.array([[0.9, 0.1]], dtype=np.float32) 
# [Dato Conosciuto = 0.9, Dato Essenziale = 0.1]

# Scenario B: Alto Dato Essenziale, Basso Dato Conosciuto (Dovrebbe dare un risultato ALTO)
data_B = np.array([[0.1, 0.9]], dtype=np.float32)
# [Dato Conosciuto = 0.1, Dato Essenziale = 0.9]

# 3. Esegui la Predizione
def run_model(data):
    """Esegue l'algoritmo sul set di dati fornito."""
    result = session.run([output_name], {input_name: data})
    return result[0][0][0]

# Esecuzione degli scenari
result_A = run_model(data_A)
result_B = run_model(data_B)

# 4. Stampa dei Risultati e Analisi Filosofica
print("\n--- Risultati dell'Algoritmo Fratao ---")

print(f"Scenario A (Tanto CONOSCIUTO, poco ESSENZIALE):")
print(f"  Input: [Conosciuto: 0.9, Essenziale: 0.1] -> Risultato Predizione: {result_A:.4f}")

print(f"Scenario B (Poco CONOSCIUTO, tanto ESSENZIALE):")
print(f"  Input: [Conosciuto: 0.1, Essenziale: 0.9] -> Risultato Predizione: {result_B:.4f}")

print("\nANALISI:")
if result_B > result_A:
    print("✅ L'algoritmo ha successo! La predizione è molto più alta quando si 'Svuota del Conosciuto' (Scenario B) e si concentra sull'Essenziale.")
else:
    print("❌ L'algoritmo non ha funzionato come previsto.")
