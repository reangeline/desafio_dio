import requests
from bs4 import BeautifulSoup
from langchain_openai.chat_models.azure import AzureChatOpenAI

# Configurações do Azure OpenAI
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
URL_TO_TRANSLATE=""
TARGET_LANGUAGE=""

def scrape_and_translate(url: str, target_language: str) -> str:
    """
    Faz o scraping de uma URL e traduz o conteúdo para o idioma de destino.
    
    :param url: A URL do artigo ou página web a ser traduzida.
    :param target_language: O idioma de destino (ex: 'portuguese', 'spanish').
    :return: O texto traduzido.
    """
    # Fazendo o scraping do conteúdo
    response = requests.get(url)
    if response.status_code != 200:
        return f"Erro ao acessar a URL: {response.status_code}"
    
    soup = BeautifulSoup(response.text, "html.parser")
    # Extraindo texto visível (ajuste de acordo com o site)
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    
    if not text.strip():
        return "Não foi possível encontrar conteúdo textual na página."
    
    # Configurando o modelo OpenAI via Azure
    chat = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name="gpt-4o-mini", 
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2023-03-15-preview",
        max_retries=0
    )
    
    messages = [
        (
            # ("system", "Voce atua como um tradutor de textos"),
            ("human", f"Traduza o seguinte texto para {target_language}:\n{text}")
        )]

    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"Erro ao traduzir o texto: {str(e)}"

# Exemplo de uso
if __name__ == "__main__":
    url = URL_TO_TRANSLATE
    target_language = TARGET_LANGUAGE
    translated_text = scrape_and_translate(url, target_language)
    print(translated_text)
