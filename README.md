# REASON
REASON – Retrieval-enhanced Evidence Assessment with Synthesis and Organized Narration: An End-to-end agent-based LLM pipeline for evidence synthesis using retrieval, classification, and structured summarization.

## 🛠️ Development Workflow

### 🧪 Primary Development

- Use **Google Colab** notebooks (Enterprise or local) for rapid experimentation, prototyping, and testing.
- All code changes should be committed to your own development branches via GitHub and a PR for `main` branch should be created for merging code with `main` since it acts as the **single source of truth** for both Colab and VM environments.

---

### ☁️ Development & Running on GCP VMs (For Large-Scale Training / Testing)

We maintain a **custom GCP image** that includes Python, Git, virtualenv, and all REASON dependencies preinstalled.

#### 🔄 Current Base Image
```reason-v1-2025-06-13```

> 💡 We recommend using the format: `reason-v<major-version>-<yyyy>-<mm>-<dd>` for image versioning.

---

## 🚀 Steps to Spin Up a New VM from the Custom Image

#### 1. Go to:
[Google Cloud Console → Compute Engine → VM Instances](https://console.cloud.google.com/compute/instances)

#### 2. Click **“Create Instance”**

#### 3. Configure your instance as needed:
- **Name**: `reason-dev`, `reason-train`, etc.
- **Region & Zone**: Choose `us-east-4 (North Virginia)`, Zone `any` or as preferred.
- **Machine Type**: Choose based on your workload (e.g., `e2-standard-4`, `n2-highmem-8`, etc.)

#### 4. Under **Operating system and storage section on the left**:
- Click **Change**
- Go to the **Custom Images** tab
- Select `reason-v1-2025-06-13` image or other version as preferred

#### 5. Optionally change boot disk size (recommended 50GB+)

#### 6. Complete any other configs that you want to

#### 7. Click **Create**

Now your VM should be up and running and it should already be setup with the environment dependencies needed for REASON. It will also have REASON repository present already under `workspace/REASON` folder. 

## 🔑 Setting Up SSH Keys for GitHub (One-Time on Each VM)

When you SSH into a newly created VM for the first time, you will need to set up **SSH keys** so the VM can authenticate with GitHub for `git pull`, `git push`, and other operations.

Follow these steps on the VM:

---

### 🧾 1. Generate an SSH Key Pair

Replace the email below with your own GitHub email:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```
When prompted for a file path, press Enter to accept the default (~/.ssh/id_ed25519)

Optionally set a passphrase (or leave it blank)

### 📋 2. Copy the Public Key
```bash
cat ~/.ssh/id_ed25519.pub
```
This will output a single-line public key. Copy it to your clipboard.

### 🔗 3. Add the SSH Key to GitHub
Go to https://github.com/settings/keys

Click "New SSH key"

Paste the copied public key into the textbox

Give it a descriptive title (e.g., GCP Reason VM)

Click Add SSH key

### ✅ 4. Test the Connection to GitHub

Run:

```bash
ssh -T git@github.com
```

If everything is working, you should see a message like:

```Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.```

That's it, your VM can now securely pull and push to GitHub repos using SSH.

### ⚠️⚠️ Read the following Sections to Maintain VMs properly ⚠️⚠️

## 🔁 Keeping Your VM Updated

#### ⚠️ **Remember:** Every time you log in or SSH into a VM, follow the steps below to make sure your code and environment are up to date.

---

### 🪜 Step-by-Step Process

#### ✅ 1. Pull the latest code from GitHub
```bash
cd ~/workspace/REASON
git pull origin main
```

> This ensures you have the most recent code, notebooks, and updates from Colab or other contributors.

---

#### ✅ 2. Activate the Python virtual environment
If you're inside the `REASON` folder, run:
```bash
source reason_py_env/bin/activate
```

> This activates the `venv` created during the initial VM setup.

---

#### ✅ 3. Install any new dependencies
```bash
pip install -r requirements.txt
```

> This makes sure your environment has all the latest packages needed to run the project.

---

Following these three steps ensures your VM is always running the latest stable version of REASON and won't break due to missing packages or stale code.


## 🧹 Deleting / Stopping VMs
To save credits:

Stop the VM when not in use

Use custom image again to resume later with larger or smaller configs

## 📦 Dependencies
All packages are pre-installed on the image.
To upgrade locally or on VM:

```bash
pip install -r requirements.txt
```

## 📂 Project Directory Structure on VM

```bash
~/workspace/
  └── REASON/
       ├── reason_py_env/       # Python virtual environment (created on VM)
       ├── requirements.txt     # All dependencies listed here
       ├── env-setup/           
       │    └── setup-reason-vm.sh  # Bash script to set up environment locally if required
       ├── notebooks/                            
       └── ...
```

# Have fun synthesizing! 🧠✨
