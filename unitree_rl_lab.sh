#!/usr/bin/env bash

export UNITREE_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python_exe=""
env_type=""

if [[ -n "${CONDA_PREFIX}" ]]; then
    python_exe="${CONDA_PREFIX}/bin/python"
    env_type="conda"
elif [[ -n "${VIRTUAL_ENV}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
    python_exe="${VIRTUAL_ENV}/bin/python"
    env_type="venv"
elif [[ -x "${UNITREE_RL_LAB_PATH}/.venv/bin/python" ]]; then
    python_exe="${UNITREE_RL_LAB_PATH}/.venv/bin/python"
    env_type="uv"
else
    echo "[Error] No Conda or virtual environment detected."
    echo "        Activate your environment first, or create a local uv environment with 'uv venv'."
fi


# task env name autocomplete
_ut_rl_lab_python_argcomplete_wrapper() {
    if [[ -z "${python_exe}" ]]; then
        return 0
    fi

    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./unitree_rl_lab.sh"


_ut_setup_conda_env() {

    # copied from isaaclab/_isaac_sim/setup_conda_env.sh
    # add source unitree_rl_lab.sh to conda activate.d
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' \
        '# for unitree_rl_lab' \
        'source '${UNITREE_RL_LAB_PATH}'/unitree_rl_lab.sh' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup conda variables to load the binaries
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi
}

_ut_install() {
    git lfs install # ensure git lfs is installed

    case "${env_type}" in
        conda)
            pip install -e "${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/"
            _ut_setup_conda_env
            ;;
        uv|venv)
            if ! command -v uv > /dev/null 2>&1; then
                echo "[Error] 'uv' is required for virtual environment installs."
                exit 1
            fi
            uv pip install -e "${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/"
            ;;
        *)
            echo "[Error] Cannot install because no supported Python environment was detected."
            exit 1
            ;;
    esac

    if command -v activate-global-python-argcomplete > /dev/null 2>&1; then
        activate-global-python-argcomplete
    fi
}

_ut_require_python() {
    if [[ -z "${python_exe}" ]]; then
        echo "[Error] No Python environment available for this command."
        exit 1
    fi
}

# pass the arguments
case "$1" in
    -i|--install)
        _ut_install
        ;;
    -l|--list)
        shift
        _ut_require_python
        "${python_exe}" "${UNITREE_RL_LAB_PATH}/scripts/list_envs.py" "$@"
        ;;
    -p|--play)
        shift
        _ut_require_python
        "${python_exe}" "${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/play.py" "$@"
        ;;
    -t|--train)
        shift
        _ut_require_python
        "${python_exe}" "${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py" --headless "$@"
        ;;
    *) # unknown option
        ;;
esac
