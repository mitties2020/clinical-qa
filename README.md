# clinical-qa

## Deployment troubleshooting (Render + PowerShell)

If deploys are not reflecting latest changes, work through the checks below in order.

## Windows fix: PowerShell commands not working

If PowerShell shows:

```powershell
git : The term 'git' is not recognized as the name of a cmdlet...
```

Git is not installed yet, or PowerShell has not reloaded your PATH.

### 1) Install Git for Windows (pick one)

```powershell
winget install --id Git.Git -e --source winget
```

Or install from: https://git-scm.com/download/win

### 2) Restart terminal
Close PowerShell completely, then open a new PowerShell window.

### 3) Verify Git works
Run these commands one line at a time:

```powershell
where.exe git
git --version
```

If `where.exe git` returns nothing, reboot Windows once and retry.

### 4) Go to your project folder

```powershell
cd C:\path\to\clinical-qa
```

Tip: in File Explorer, open the project folder, click the address bar, type `powershell`, and press Enter.

### 5) Push your updates to the branch Render deploys

```powershell
git checkout work
git checkout -B main
git push -u origin main
```

If this is your first push from this machine, set remote first:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 6) Deploy on Render
- Open your Render service.
- Confirm Branch is `main`.
- Click **Manual Deploy → Deploy latest commit**.

## No-terminal fallback (if you still cannot run commands)
1. Open the repo on GitHub in your browser.
2. Use **Add file → Upload files**.
3. Upload changed files and commit directly to `main`.
4. In Render, click **Manual Deploy → Deploy latest commit**.

## Quick Render checklist
- Service branch is `main`.
- Latest Render deploy SHA matches GitHub `main` SHA.
- Required environment variables are present in Render.

## What to do now (quick path)
1. Install Git (if `git` is still not recognized):
   - `winget install --id Git.Git -e --source winget`
2. Open a **new** PowerShell window.
3. Check Git works:
   - `git --version`
4. Go to your project folder:
   - `cd C:\path\to\clinical-qa`
5. Push your changes to `main`:
   - `git checkout work`
   - `git checkout -B main`
   - `git push -u origin main`
6. Deploy in Render:
   - Service branch = `main`
   - **Manual Deploy → Deploy latest commit**
7. Confirm success:
   - Render shows latest GitHub commit SHA
   - Website reflects your updates after hard refresh (`Ctrl + F5`)

If any command fails, copy the exact error and run:
- `where.exe git`
- `git remote -v`
- `git branch --show-current`

