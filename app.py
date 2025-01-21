from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Charger le modèle et le tokenizer fine-tunés
checkpoint_path = "checkpoint-324"  # Assurez-vous que le chemin est correct et contient tous les fichiers nécessaires
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

# Fonction pour générer du texte à partir d'un prompt
def generate_text(prompt):
    # Ajouter un préfixe explicite pour guider la réponse
    prompt_with_instruction = f"Answer the following question in a detailed manner: {prompt}"

    # Encoder le texte d'entrée
    inputs = tokenizer.encode(prompt_with_instruction, return_tensors="pt")

    # Générer le texte avec des paramètres plus précis pour éviter la répétition
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            num_return_sequences=1,  # Générer une seule séquence
            no_repeat_ngram_size=2,  # Éviter la répétition de n-grams
            temperature=0.9,         # Ajuster la créativité du texte
            top_k=50,                # Limiter les choix possibles à 50
            top_p=0.95,              # Utiliser un échantillonnage nucleus (top-p)
            do_sample=True,          # Activer l'échantillonnage
            eos_token_id=tokenizer.eos_token_id,  # Arrêter à la fin de la séquence
            pad_token_id=tokenizer.pad_token_id,  # Utiliser un token de padding
            max_length=150,          # Limiter la longueur du texte généré pour éviter des réponses trop courtes
            min_length=50,           # Minimum pour avoir une réponse complète
        )

    # Décoder le texte généré
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Supprimer le prompt du texte généré
    if generated_text.startswith(prompt_with_instruction):
        generated_text = generated_text[len(prompt_with_instruction):].strip()

    # Vérifier si la réponse commence par "Answer:" et ajuster si nécessaire
    if not generated_text.startswith("Answer:"):
        generated_text = "Answer: " + generated_text

    # Supprimer les parties indésirables comme les questions répétées
    if "Question:" in generated_text:
        generated_text = generated_text.split("Question:")[0].strip()

    return generated_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['Text']
    
    # Générer le texte sans spécifier max_length ou min_length
    generated_text = generate_text(text)
    
    # Nettoyer et formater la sortie
    output = f"<strong>Question:</strong> {text}<br><br><strong>Generated Text:</strong> {generated_text}"
    
    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
