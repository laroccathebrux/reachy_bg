# Claude Code Environment Setup Prompt

**Cole este texto INTEIRO no Claude Code para montar o ambiente automaticamente**

---

## 📋 PROMPT PARA COLAR NO CLAUDE CODE

```
Você vai montar o ambiente de desenvolvimento para o projeto Eldritch Horror + Reachy.

REPOSITÓRIO: https://github.com/laroccathebrux/reachy_bg
BRANCH: main
DIRETÓRIO DE TRABALHO: /home/claude/eldritch-horror-reachy

TAREFAS (nesta ordem):

### 1. Verificar Estrutura do Projeto
- Confirme que existem: src/, docs/, requirements.txt, setup.py, .env.example
- Listar todos os arquivos: find . -type f ! -path './.git/*' | sort
- Mostrar contagem total de arquivos

### 2. Setup do Python Virtual Environment
- Navegar para /home/claude/eldritch-horror-reachy
- Criar venv: python3 -m venv venv
- Ativar: source venv/bin/activate
- Verificar Python: python --version (deve ser 3.9+)
- Upgrade pip: pip install --upgrade pip

### 3. Instalar Dependências
- pip install -r requirements.txt
- Aguarde conclusão (pode levar alguns minutos)
- Mostre: pip list | tail -20

### 4. Instalar em Modo Desenvolvimento (opcional)
- pip install -e ".[dev]"

### 5. Teste de Imports Críticos
Execute cada teste separadamente e mostre resultado:

```bash
python -c "import cv2; print('✅ OpenCV OK')"
python -c "import torch; print('✅ PyTorch OK')"
python -c "import torchvision; print('✅ TorchVision OK')"
python -c "import qdrant_client; print('✅ Qdrant Client OK')"
python -c "from src.config import OLLAMA_MODEL; print(f'✅ Config OK: {OLLAMA_MODEL}')"
python -c "from src.logger import get_logger; logger = get_logger('test'); logger.info('✅ Logger OK')"
python -c "from src import get_logger; print('✅ Package Exports OK')"
```

### 6. Configurar .env
- Copiar: cp .env.example .env
- Exibir conteúdo de .env.example: cat .env.example
- Exibir conteúdo de .env criado: cat .env
- Avisar ao usuário: "Edite .env com suas credenciais (API keys, hosts, etc) antes de rodar código"

### 7. Verificar Estrutura de Módulos
- Listar estrutura src/: ls -la src/*/
- Contar arquivos Python: find src/ -name "*.py" | wc -l
- Contar linhas de código: find src/ -name "*.py" -exec wc -l {} + | tail -1

### 8. Verificar Git Status
- Confirme branch: git branch
- Mostre últimos commits: git log --oneline -5
- Status: git status

### 9. Teste de Estrutura do Projeto
- Verifique se pode importar src como pacote: python -c "import sys; sys.path.insert(0, '.'); from src.config import QDRANT_HOST; print(f'✅ QDRANT_HOST={QDRANT_HOST}')"
- Teste logger com contexto: python -c "from src.logger import StructuredLogger; log = StructuredLogger('test', {'phase': '0'}); log.info('Test message'); print('✅ StructuredLogger OK')"

### 10. Resumo Final

Exiba:
1. Versão do Python
2. Versões críticas de pacotes:
   ```bash
   pip list | grep -E "torch|opencv|ollama|qdrant|transformers|elevenlabs|pillow"
   ```
3. Confirmação de .env presente
4. Total de commits no repositório: git log --oneline | wc -l
5. Tamanho total do projeto: du -sh .

### 11. Próximos Passos

Imprima estas instruções:

```
✅ AMBIENTE CONFIGURADO COM SUCESSO!

📦 PRÓXIMOS PASSOS:

1. Edite .env com suas credenciais:
   - ELEVENLABS_API_KEY=seu_token_aqui (opcional por enquanto)
   - Outros valores já estão configurados para localhost

2. Verifique que Ollama e Qdrant vão rodar localmente:
   - Terminal 1: ollama serve
   - Terminal 2: docker run -p 6333:6333 qdrant/qdrant

3. Leia a documentação (em ordem):
   - PROJECT_STATUS.md
   - GETTING_STARTED.md
   - DESIGN_DOCUMENT.md
   - PHYSICAL_SETUP.md

4. Para começar Phase 1 (Vision), peça:
   "Cria Phase 1.1: Board Capture Module"

📍 LOCALIZAÇÃO DO PROJETO: /home/claude/eldritch-horror-reachy
📍 REPOSITÓRIO: https://github.com/laroccathebrux/reachy_bg
🔗 BRANCH: main
```

SE HOUVER ERROS:
- Mostre o erro completo
- Sugira solução (ex: versão Python incompatível, falta de espaço, etc)
- Tente resolver automaticamente se possível
```

---

## 🎯 Como Usar Este Arquivo

1. **Abra Claude Code**
2. **Crie uma nova conversa**
3. **Cole todo o conteúdo acima** (entre os ```triple backticks```)
4. **Envie a mensagem**

Claude Code vai executar automaticamente todos os passos e montar o ambiente completo.

---

## ✅ Resultado Esperado

Após rodar o prompt, você deve ver:

```
✅ OpenCV OK
✅ PyTorch OK
✅ TorchVision OK
✅ Qdrant Client OK
✅ Config OK: qwen2.5:72b
✅ Logger OK
✅ Package Exports OK
✅ StructuredLogger OK

Python 3.11.x (ou superior)

Versões críticas:
torch==2.1.0
opencv-python==4.8.0
qdrant-client==2.7.0
...

✅ AMBIENTE CONFIGURADO COM SUCESSO!

PRÓXIMOS PASSOS:
1. Edite .env com suas credenciais
2. Inicie Ollama: ollama serve
3. Inicie Qdrant: docker run -p 6333:6333 qdrant/qdrant
4. Solicite Phase 1: Vision Module
```

---

## 💡 Se Algo Der Errado

Peça ao Claude Code:

```
Resolve o erro de [descrição] e continua o setup
```

Ele vai diagnosticar e corrigir automaticamente.

---

## 📂 Arquivo Gerado

Este arquivo está em: `/home/claude/eldritch-horror-reachy/CLAUDE_CODE_SETUP_PROMPT.md`

Você pode:
- Copiar o prompt deste arquivo
- Usar em qualquer momento
- Compartilhar com outros desenvolvedores
