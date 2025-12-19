#!/bin/bash
# ============================================
# verl训练环境一键配置脚本
# ============================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   verl 训练环境配置脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检测系统
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac
echo -e "${BLUE}检测到操作系统: $PLATFORM${NC}"

# ============================================
# 步骤1: 检查Python
# ============================================
echo ""
echo -e "${YELLOW}[1/6] 检查Python环境...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3，请先安装Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Python版本: $PYTHON_VERSION${NC}"

# 检查版本
if [[ "$(echo "$PYTHON_VERSION < 3.10" | bc)" -eq 1 ]]; then
    echo -e "${RED}错误: 需要Python 3.10+，当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi

# ============================================
# 步骤2: 检查CUDA
# ============================================
echo ""
echo -e "${YELLOW}[2/6] 检查CUDA环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}GPU: $GPU_NAME${NC}"
    echo -e "${GREEN}GPU数量: $GPU_COUNT${NC}"
    echo -e "${GREEN}驱动版本: $CUDA_VERSION${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}警告: 未检测到NVIDIA GPU，将以CPU模式安装${NC}"
    HAS_GPU=false
fi

# ============================================
# 步骤3: 创建/激活虚拟环境
# ============================================
echo ""
echo -e "${YELLOW}[3/6] 配置虚拟环境...${NC}"

# 检查是否在conda环境中
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${GREEN}检测到conda环境: $CONDA_DEFAULT_ENV${NC}"
    VENV_TYPE="conda"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}检测到venv环境: $VIRTUAL_ENV${NC}"
    VENV_TYPE="venv"
else
    echo -e "${YELLOW}未检测到虚拟环境${NC}"
    echo ""
    echo "是否创建新的conda环境? (y/n)"
    read -r CREATE_ENV

    if [[ "$CREATE_ENV" == "y" || "$CREATE_ENV" == "Y" ]]; then
        if command -v conda &> /dev/null; then
            echo "创建conda环境 'verl'..."
            conda create -n verl python=3.10 -y
            echo -e "${GREEN}环境创建成功！请运行以下命令激活环境后重新运行此脚本:${NC}"
            echo -e "${BLUE}  conda activate verl${NC}"
            echo -e "${BLUE}  ./setup_env.sh${NC}"
            exit 0
        else
            echo -e "${YELLOW}未找到conda，创建venv环境...${NC}"
            python3 -m venv .venv
            echo -e "${GREEN}请运行以下命令激活环境后重新运行此脚本:${NC}"
            echo -e "${BLUE}  source .venv/bin/activate${NC}"
            echo -e "${BLUE}  ./setup_env.sh${NC}"
            exit 0
        fi
    fi
fi

# ============================================
# 步骤4: 安装PyTorch
# ============================================
echo ""
echo -e "${YELLOW}[4/6] 安装PyTorch...${NC}"

# 检查是否已安装
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'CPU')")
    echo -e "${GREEN}PyTorch已安装: $TORCH_VERSION (CUDA: $TORCH_CUDA)${NC}"
else
    echo "安装PyTorch..."
    if [[ "$HAS_GPU" == true ]]; then
        # 检测CUDA版本并安装对应PyTorch
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

        if [[ "$CUDA_MAJOR" -ge 12 ]]; then
            pip install torch --index-url https://download.pytorch.org/whl/cu121
        else
            pip install torch --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        pip install torch
    fi
fi

# ============================================
# 步骤5: 安装verl和依赖
# ============================================
echo ""
echo -e "${YELLOW}[5/6] 安装verl和依赖...${NC}"

# 安装基础依赖
pip install --upgrade pip

# 安装verl
if python3 -c "import verl" 2>/dev/null; then
    VERL_VERSION=$(python3 -c "import verl; print(verl.__version__)")
    echo -e "${GREEN}verl已安装: $VERL_VERSION${NC}"
else
    echo "安装verl..."
    pip install verl
fi

# 安装推理引擎
echo "安装vLLM推理引擎..."
pip install vllm || echo -e "${YELLOW}vLLM安装失败，可稍后手动安装${NC}"

# 安装其他依赖
echo "安装其他依赖..."
pip install -r requirements.txt

# ============================================
# 步骤6: 验证安装
# ============================================
echo ""
echo -e "${YELLOW}[6/6] 验证安装...${NC}"

echo ""
echo "检查关键包..."

check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        VERSION=$(python3 -c "import $1; print(getattr($1, '__version__', 'unknown'))")
        echo -e "  ${GREEN}✓${NC} $1: $VERSION"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1: 未安装"
        return 1
    fi
}

check_package torch
check_package transformers
check_package verl
check_package ray
check_package vllm || true
check_package gradio
check_package pandas

# ============================================
# 完成
# ============================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   环境配置完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "下一步:"
echo -e "  1. 准备训练数据"
echo -e "  2. 选择训练方法并运行:"
echo -e "     ${BLUE}cd sft && ./train.sh${NC}"
echo -e "     ${BLUE}cd grpo && ./train.sh${NC}"
echo -e "     ${BLUE}cd ppo && ./train.sh${NC}"
echo -e "     ${BLUE}cd dpo && ./train.sh${NC}"
echo -e "     ${BLUE}cd gspo && ./train.sh${NC}"
echo ""
echo -e "  3. 启动监控面板:"
echo -e "     ${BLUE}cd monitor && python app.py${NC}"
echo ""

# 设置脚本执行权限
chmod +x sft/train.sh ppo/train.sh grpo/train.sh dpo/train.sh gspo/train.sh 2>/dev/null || true

echo -e "${GREEN}Happy Training!${NC}"
