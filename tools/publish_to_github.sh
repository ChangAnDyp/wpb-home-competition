#!/usr/bin/env bash
# 把比赛仓库推送到 GitHub
#
# 用法（二选一）：
#   1) SSH 方式（推荐，配置一次后永久免密）
#      bash ~/catkin_ws/competition/publish_to_github.sh git@github.com:你的用户名/仓库名.git
#
#   2) HTTPS + 令牌方式
#      bash ~/catkin_ws/competition/publish_to_github.sh https://<令牌>@github.com/你的用户名/仓库名.git
#
# 说明：令牌不要直接写进 shell 历史的话，可以先 export：
#      export GH_TOKEN=xxxx
#      bash ~/catkin_ws/competition/publish_to_github.sh github.com/你的用户名/仓库名

set -eu

REPO_DIR="${REPO_DIR:-$HOME/wpb_competition_repo}"
BRANCH="${BRANCH:-master}"

say() { printf '\033[1;36m>>>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m###\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$REPO_DIR/.git" ] || die "找不到 git 仓库：$REPO_DIR"
cd "$REPO_DIR"

arg="${1:-}"
[ -n "$arg" ] || die "请给出仓库地址，例如 git@github.com:user/repo.git"

# 支持只写 github.com/user/repo 的形式，自动补全
case "$arg" in
  git@*|https://*|http://*|ssh://*) url="$arg" ;;
  github.com/*)
      if [ -n "${GH_TOKEN:-}" ]; then
          url="https://${GH_TOKEN}@${arg}.git"
      else
          url="https://${arg}.git"
      fi
      ;;
  *) die "无法识别的地址：$arg" ;;
esac

say "仓库目录：$REPO_DIR"
say "分支：$BRANCH"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$url"
    say "已更新 origin"
else
    git remote add origin "$url"
    say "已添加 origin"
fi

# 不要把人脸照片、注册档案推上去
leak=$(git ls-files | grep -cE "data/owner/.*\.(jpg|jpeg|png|webp)$|owner_profile|\.npz$" || true)
if [ "$leak" != "0" ]; then
    die "检测到 $leak 个疑似隐私文件已被纳入版本控制，请先处理（git rm --cached）"
fi
say "隐私检查通过（无照片/注册档案）"

say "开始推送（首次可能需要授权）"
if git push -u origin "$BRANCH"; then
    say "推送成功"
    git remote get-url origin | sed 's#//[^@]*@#//***@#' | sed 's#^#    远程地址: #'
else
    die "推送失败。若提示认证失败：SSH 方式请确认公钥已加到 GitHub；HTTPS 方式请确认令牌有效且勾选了 repo 权限"
fi
