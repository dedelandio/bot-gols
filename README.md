# Bot de Alertas de Gols no Telegram (Poisson + Value)

Este pacote já contém tudo pronto para você subir na nuvem (Railway) **sem precisar copiar do canvas**.

## 0) O que tem na pasta
- `bot_gols.py` — o código do bot
- `requirements.txt` — dependências
- `Procfile` — define que é um worker 24/7
- `runtime.txt` — fixa a versão do Python (opcional)
- `README.md` — este guia

---

## 1) Telegram: criar o bot e IDs (1x)
1. Fale com **@BotFather** → `/newbot` → pegue o **TOKEN** (formato `123456:ABC...`).  
2. Envie uma mensagem para o seu bot.  
3. Pegue seu **CHAT_ID** com **@userinfobot** (para pessoa) ou adicione o bot como **Admin** no seu **canal** e obtenha o ID (geralmente começa com `-100`).

Você vai precisar de:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

---

## 2) Subir para o GitHub
1. Crie um repositório novo no GitHub (ex.: `bot-gols`).  
2. Faça upload destes arquivos para o repositório (pelo site ou via `git`).

---

## 3) Railway (nuvem 24/7)
1. Entre em **railway.app** (login com GitHub).  
2. **New Project → Deploy from GitHub** → selecione seu repositório.  
3. Após o build, abra o serviço e vá em **Variables** e adicione:
   - `TELEGRAM_BOT_TOKEN` = seu token
   - `TELEGRAM_CHAT_ID` = seu chat id
   - (opcional) `FOOTBALL_DATA_TOKEN` = token da football-data.org
   - (opcional) `LEAGUE_CODES` = `BSA,SA,PD,PL,BL1,FL1` (exemplo)

> O `Procfile` já está configurado como: `worker: python bot_gols.py`  
> Isso roda o bot em background de forma contínua.

### Teste imediato (execução única)
Se quiser forçar um teste agora:  
- Edite o `Procfile` no GitHub para: `worker: python bot_gols.py --once`  
- A Railway vai redeployar; veja os **Logs** e confirme a mensagem no Telegram.  
- Depois **volte** para `worker: python bot_gols.py` para rodar 24/7.

---

## 4) Onde você travou (resolvido)
Antes você precisaria **copiar o arquivo do canvas** para criar `bot_gols.py`.  
**Agora não precisa**: este pacote já traz o arquivo pronto.  
Basta **baixar o ZIP**, extrair a pasta e subir os arquivos para o GitHub.

---

## 5) Opcional: rodar local
Crie um `.env` com:
```
TELEGRAM_BOT_TOKEN=xxxxx:yyyyy
TELEGRAM_CHAT_ID=123456789
```
Instale dependências:
```
pip install -r requirements.txt
```
Teste:
```
python bot_gols.py --once
```
Rodar contínuo:
```
python bot_gols.py
```

---

## 6) Ajustes
- Intervalo (padrão 30 min): ajustar no `@scheduler.scheduled_job('interval', minutes=30)`
- Ligas: `LEAGUE_CODES`
- Critérios de envio: `margin_value`, `min_edge`, `min_prob` dentro de `build_signals()`

Boa sorte e bons sinais! ⚽️
