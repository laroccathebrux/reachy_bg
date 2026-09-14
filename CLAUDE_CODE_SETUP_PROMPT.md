# Claude Code Setup Prompt - Versão Final Simplificada

**Cole este texto NO CLAUDE CODE para começar**

---

## 📌 Workspace Único

**Diretório do Projeto** (seu Mac):  
```
/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/
```

**Repositório**: https://github.com/laroccathebrux/reachy_bg

**Workflow**: 
- Você trabalha no seu Mac
- Claude Code aponta para esse mesmo diretório
- Mudanças feitas direto lá
- Git push/pull sincroniza tudo

---

## 📋 PROMPT PARA COLAR NO CLAUDE CODE

```
Você vai trabalhar no projeto Eldritch Horror + Reachy.

REPOSITÓRIO: https://github.com/laroccathebrux/reachy_bg
WORKSPACE: /Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg

TAREFAS SEMPRE (nesta ordem):

### 1. Navegar para o Workspace
cd /Users/alessandrolaroccasilveira/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
pwd  # Confirme que está no diretório correto

### 2. Sincronizar com GitHub
git pull origin main
git status  # Verifique estado

### 3. Verificar Ambiente Python
source venv/bin/activate  # Ativar ambiente virtual
python --version  # Deve ser 3.9+
python -c "from src.config import OLLAMA_MODEL; print(f'✅ Config OK: {OLLAMA_MODEL}')"

### 4. Executar Tarefas Solicitadas
[Aqui você fará as mudanças/features que eu pedir]
- Editar arquivos
- Criar módulos
- Escrever testes
- Etc

### 5. Testar Localmente
pytest  # Rodar testes
# Ou testes específicos: pytest tests/test_x.py

### 6. Commit e Push
git add .
git commit -m "Feature: descrição clara da mudança"
git push origin main

### 7. Confirmação Final
git log --oneline -3  # Mostrar últimos commits
git status  # Deve estar clean
```

---

## 🎯 Como Usar

### Primeira Vez (Setup Inicial)

1. Abra Claude Code
2. Crie uma conversa nova
3. Cole TODO o texto acima (entre os triple backticks)
4. Envie

Claude Code vai verificar o ambiente e confirmar que está pronto.

### Próximas Vezes (Desenvolvimento Diário)

Apenas peça:

```
Sincroniza e faz:
[descrição da tarefa]
```

Claude Code vai:
1. Pull do GitHub (naquele diretório)
2. Fazer a tarefa
3. Testar
4. Push para GitHub

Você no seu Mac depois faz:
```bash
git pull origin main
```

---

## ✅ O Que Vai Acontecer

```
Você (seu Mac):
  ├─ git pull origin main
  ├─ Desenvolve localmente
  └─ git push origin main

Claude Code (este chat):
  ├─ cd [seu diretório]
  ├─ git pull origin main
  ├─ Faz mudanças/cria features
  ├─ pytest (testa)
  └─ git push origin main

GitHub:
  └─ Sincroniza mudanças entre você e Claude Code
```

---

## 📝 Exemplo de Uso

**Você pede:**
```
Sincroniza e cria Phase 1.1: Board Capture Module com testes e documentação
```

**Claude Code faz:**
```bash
cd /Users/alessandrolaroccasilveira/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
git pull origin main
# ... cria src/vision/board_capture.py
# ... cria tests/test_board_capture.py
pytest
git add .
git commit -m "Feature: Phase 1.1 Board Capture Module"
git push origin main
```

**Você no seu Mac recebe:**
```bash
git pull origin main
# Arquivos aparecem localmente
```

---

## 🔄 Fluxo Diário

1. **Seu Mac (começa):**
   ```bash
   git pull origin main
   ```

2. **Você pede ao Claude Code:**
   ```
   Sincroniza e cria [feature]
   ```

3. **Claude Code executa** (naquele diretório)

4. **Seu Mac recebe:**
   ```bash
   git pull origin main
   ```

5. **Repete conforme necessário**

---

## 📍 Localização Final

- **Workspace Único**: `/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/`
- **Repositório**: https://github.com/laroccathebrux/reachy_bg
- **Branch**: main
- **Claude Code**: Aponta para esse mesmo diretório
- **Sincronização**: Via GitHub

---

## 💾 Histórico

- v1.2: Simplificado - Um único diretório no Mac
- Claude Code trabalha DIRETO no seu diretório
- Sem cópias, sem confusão, sem `/tmp/`

---

**Tá pronto! Cole no Claude Code agora!** 🚀
```
