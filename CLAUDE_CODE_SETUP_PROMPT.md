# Claude Code Environment Setup Prompt

**Cole este texto INTEIRO no Claude Code para montar o ambiente automaticamente**

---

## 📌 NOVO WORKFLOW (Atualizado)

**Repositório Único**: https://github.com/laroccathebrux/reachy_bg  
**Workspace Principal (Seu Mac)**: `/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/`

**Como funciona:**
1. Você trabalha no seu Mac (dev iterativo, testes, commit/push)
2. Você solicita features ao Claude Code via este chat
3. Claude Code puxa latest do GitHub, faz mudanças, testa, push
4. Você faz pull no seu Mac (recebe mudanças)

---

## 📋 PROMPT PARA COLAR NO CLAUDE CODE

```
Você vai montar o ambiente de desenvolvimento para o projeto Eldritch Horror + Reachy.

REPOSITÓRIO: https://github.com/laroccathebrux/reachy_bg
BRANCH: main
WORKSPACE: Usar repositório do GitHub (sem cópia separada local)

TAREFAS (nesta ordem):

### 1. Clonar ou Atualizar Repositório
- git clone https://github.com/laroccathebrux/reachy_bg.git /tmp/reachy_bg_work
- cd /tmp/reachy_bg_work
- Confirme que existem: src/, docs/, requirements.txt, setup.py, .env.example
- Mostrar contagem total de arquivos: find . -type f ! -path './.git/*' | wc -l

### 2. Setup do Python Virtual Environment
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

📍 WORKSPACE (SEU MAC): ~/Documents/Documents - USSILVEIRAAXWR2/reachy_bg
📍 REPOSITÓRIO ÚNICO: https://github.com/laroccathebrux/reachy_bg
🔗 BRANCH: main
```

### 12. Sincronização com GitHub

Imprima instruções de sincronização:

```
🔄 SINCRONIZAÇÃO COM GITHUB:

ANTES DE COMEÇAR QUALQUER TAREFA:
git pull origin main

DEPOIS DE TERMINAR UMA TAREFA:
git add .
git commit -m "Feature/Fix: descrição clara"
git push origin main

SEU MAC RECEBE AS MUDANÇAS:
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
git pull origin main
```

SE HOUVER CONFLITOS:
- Mostre o erro completo
- Sugira solução (ex: merge conflict, versão Python incompatível, falta de espaço)
- Tente resolver automaticamente se possível
```

---

## 🔄 WORKFLOW DIÁRIO

### SEU MAC (Desenvolvimento Principal)
```bash
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
source venv/bin/activate

# Comece o dia
git pull origin main

# Faça mudanças
code src/

# Teste localmente
pytest

# Commit e push
git add .
git commit -m "Feature: descrição"
git push origin main
```

### CLAUDE CODE (Solicitações de Features)

Quando você pedir uma feature:

```bash
# 1. Claude Code clona/atualiza do GitHub
git clone https://github.com/laroccathebrux/reachy_bg.git
# ou: git pull origin main (se já tiver)

# 2. Faz mudanças conforme solicitado
# Testa localmente

# 3. Push para GitHub
git add .
git commit -m "Feature: descrição"
git push origin main

# 4. Você recebe no seu Mac
git pull origin main
```

**No seu Mac, depois que Claude Code terminar:**
```bash
git pull origin main  # Recebe mudanças
```

---

## 🎯 Como Usar Este Arquivo

### Primeira Vez (Setup Inicial)

1. **Abra Claude Code**
2. **Crie uma nova conversa**
3. **Cole todo o conteúdo abaixo** (entre os ```triple backticks```)
4. **Envie a mensagem**

Claude Code vai executar automaticamente todos os passos e montar o ambiente.

### Próximas Vezes (Desenvolvimento)

Não precisa rodar o prompt completo. Apenas peça ao Claude Code:

```
Sincroniza o repositório:
git pull origin main

Depois execute:
[descrição da tarefa]
```

Isso mantém tudo sincronizado entre seu Mac e Claude Code.

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

📍 WORKSPACE PRINCIPAL: ~/Documents/Documents - USSILVEIRAAXWR2/reachy_bg (Seu Mac)
🔗 REPOSITÓRIO: https://github.com/laroccathebrux/reachy_bg
📌 BRANCH: main

PRÓXIMOS PASSOS:

1. NO SEU MAC:
   - Edite .env com suas credenciais
   - Inicie Ollama: ollama serve
   - Inicie Qdrant: docker run -p 6333:6333 qdrant/qdrant
   - Teste: pytest

2. NO CLAUDE CODE (este chat):
   - Peça: "Cria Phase 1.1: Board Capture Module com testes"
   - Claude Code: clona/atualiza, faz mudanças, testa, push

3. SINCRONIZE APÓS MUDANÇAS:
   - No seu Mac: git pull origin main
   - Receba as mudanças automaticamente
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

Este arquivo está em:
- **GitHub**: https://github.com/laroccathebrux/reachy_bg/blob/main/CLAUDE_CODE_SETUP_PROMPT.md
- **Seu Mac**: `~/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/CLAUDE_CODE_SETUP_PROMPT.md`

Você pode:
- Copiar o prompt deste arquivo para setup inicial
- Usar em qualquer momento
- Compartilhar com outros desenvolvedores

---

## 🔗 Fluxo Diário Simplificado

**Todo dia:**

1. No seu Mac: `git pull origin main` (começa o dia, recebe mudanças do Claude Code)
2. Desenvolva/teste localmente no seu Mac
3. `git push origin main` (quando terminar suas mudanças)

**Quando precisar de nova feature:**

4. Peça ao Claude Code neste chat: "Cria [feature]"
5. Claude Code: clona/atualiza, faz mudanças, testa, push
6. No seu Mac: `git pull origin main` (recebe mudanças)
7. Repita

---

## 💾 Histórico de Atualizações

- **v1.0**: Prompt inicial para setup
- **v1.1**: Workflow com Mac como workspace principal e Claude Code para geração
- **v1.2**: SIMPLIFICADO - Remove conceito de diretório separado no Claude Code
  - Workspace único: GitHub + Mac (`~/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/`)
  - Claude Code trabalha via prompt solicitações sem cópia local permanente
  - Sincronização: Mac ← → GitHub ← → Claude Code
