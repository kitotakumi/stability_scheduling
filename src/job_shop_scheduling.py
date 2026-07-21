"""
ジョブショップスケジューリング問題の各クラスを格納するモジュール

問題定義は problems/ ディレクトリの .txt ファイル（OR-Library標準形式）から読み込む。
シナリオ（初期ガントチャート・遅延ガントチャート）は scenarios/ ディレクトリの .json ファイルで管理する。

class JobMachineTable: 問題定義とシナリオを保持するクラス
class JobMachineChild: 遺伝子をガントチャートに展開する際に用いる情報を取得する
"""

import json
import os

# problems/ と scenarios/ のデフォルトパス
# 本モジュールは src/ 配下にあるが、データ (problems/, scenarios/) は
# リポジトリルート直下に置いているため、src/ の親を基準に解決する。
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BASE_DIR)
PROBLEMS_DIR = os.path.join(_REPO_ROOT, "problems")
SCENARIOS_DIR = os.path.join(_REPO_ROOT, "scenarios")


def load_problem(problem_name, problems_dir=None):
    """問題定義ファイル (.txt) を読み込み、m_table と pt_table を返す。

    ファイル形式 (OR-Library標準形式):
        1行目: num_jobs num_machines
        以降各行: machine_id processing_time machine_id processing_time ...
        (machine_id は 0始まり)

    Returns:
        (m_table, pt_table): 機械番号テーブルと加工時間テーブル
    """
    if problems_dir is None:
        problems_dir = PROBLEMS_DIR
    filepath = os.path.join(problems_dir, f"{problem_name}.txt")
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    num_jobs, _num_machines = map(int, lines[0].split())
    m_table = []
    pt_table = []
    for i in range(1, num_jobs + 1):
        tokens = list(map(int, lines[i].split()))
        machines = []
        times = []
        for j in range(0, len(tokens), 2):
            machines.append(tokens[j])
            times.append(tokens[j + 1])
        m_table.append(machines)
        pt_table.append(times)
    return m_table, pt_table


def load_scenario(scenario_name, scenarios_dir=None):
    """シナリオファイル (.json) を読み込む。

    Returns:
        dict: {"problem", "description", "initial_gantt", "delayed_gantt"}
    """
    if scenarios_dir is None:
        scenarios_dir = SCENARIOS_DIR
    filepath = os.path.join(scenarios_dir, f"{scenario_name}.json")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_scenario(scenario_name, problem_name, description,
                  initial_gantt, delayed_gantt, scenarios_dir=None):
    """シナリオを .json ファイルに保存する。"""
    if scenarios_dir is None:
        scenarios_dir = SCENARIOS_DIR
    os.makedirs(scenarios_dir, exist_ok=True)
    data = {
        "problem": problem_name,
        "description": description,
        "initial_gantt": initial_gantt,
        "delayed_gantt": delayed_gantt,
    }
    filepath = os.path.join(scenarios_dir, f"{scenario_name}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class JobMachineTable:
    """問題定義とシナリオを保持するクラス"""

    def __init__(self, problem_name, scenario_name=None):
        """
        Parameters:
            problem_name: 問題名 (problems/ 内の .txt ファイル名、拡張子なし)
            scenario_name: シナリオ名 (scenarios/ 内の .json ファイル名、拡張子なし)
                           Noneの場合、initial_gantt()/delayed_gantt() は None を返す
        """
        self.problem_name = problem_name
        self.scenario_name = scenario_name
        self.m_table, self.pt_table = load_problem(problem_name)
        self._scenario = None
        if scenario_name is not None:
            self._scenario = load_scenario(scenario_name)

    def get_job_count(self):
        return len(self.m_table)

    def get_machine_count(self):
        return len(self.m_table[0])

    def get_machine(self, job_num, process_index):
        return self.m_table[job_num][process_index]

    def get_process_time(self, job_num, process_index):
        return self.pt_table[job_num][process_index]

    def initial_gantt(self):
        if self._scenario is None:
            return None
        return self._scenario["initial_gantt"]

    def delayed_gantt(self):
        if self._scenario is None:
            return None
        return self._scenario["delayed_gantt"]

    def get_child(self):
        return JobMachineChild(self)

    def get_child_reactive(self, fixed_gantt, reschdule_time):
        return JobMachineChild_Reactive(self, fixed_gantt, reschdule_time)


def get_jm_table(problem_name, scenario_name=None):
    """問題とシナリオを指定してJobMachineTableを取得する。

    後方互換: 旧名 "MT6_6" → problem="mt06", "MT10_10" → problem="mt10", scenario="mt10_delay60"
    """
    # 旧名との互換マッピング
    _compat = {
        "MT6_6": ("mt06", None),
        "MT10_10": ("mt10", "mt10_delay60"),
    }
    if problem_name in _compat:
        problem_name, default_scenario = _compat[problem_name]
        if scenario_name is None:
            scenario_name = default_scenario
    return JobMachineTable(problem_name, scenario_name)


# ジョブの状態を管理するクラス  遺伝子をガントチャートに展開する際に各ジョブの次の作業の情報を管理する
class JobMachineChild:
    def __init__(self, jm_parent):
        self.jm_parent = jm_parent
        # ジョブごとの次工程番号を格納する配列
        self._next_process_indexes = [0] * jm_parent.get_job_count()
        # ジョブごとの次工程が開始できる時刻の配列
        self._next_process_starts = [0] * jm_parent.get_job_count()

    # job_numのジョブの次工程の機械番号を取得
    def get_machine(self, job_num):
        process_index = self._next_process_indexes[job_num]
        return self.jm_parent.get_machine(job_num, process_index)

    # job_numのジョブの次工程の処理時間を取得
    def get_process_time(self, job_num):
        process_index = self._next_process_indexes[job_num]
        return self.jm_parent.get_process_time(job_num, process_index)

    # job_numジョブの最も早い次工程の開始時刻を取得; この時間より後前開始できない
    def get_earliest(self, job_num):
        return self._next_process_starts[job_num]

    # job_numジョブを次工程に進める。次工程は現工程の終了時刻以降に開始できる
    def set_next_earliest(self, job_num, job_end):
        self._next_process_starts[job_num] = job_end
        self._next_process_indexes[job_num] += 1


# リスケジューリング時に各ジョブが次にできる作業の情報を管理するクラス
class JobMachineChild_Reactive:
    def __init__(self, jm_parent, fixed_gantt, reschdule_time):
        self.jm_parent = jm_parent
        # ジョブごとの次工程番号を格納する配列
        next_prosess_indexes = [0] * jm_parent.get_job_count()
        # ジョブごとの次工程が開始できる時刻の配列
        next_prosess_starts = [reschdule_time] * jm_parent.get_job_count()
        # fixed_ganttから各ジョブの次の作業の開始時間やインデックスを取得する
        for machine in fixed_gantt:
            for task in machine:
                next_prosess_indexes[task[2]] += 1
                if task[1] > reschdule_time:
                    next_prosess_starts[task[2]] = task[1]
        # ジョブごとの次工程番号を格納する配列  ganttの中で出てきたそのジョブの数が次の工程番号
        self._next_process_indexes = next_prosess_indexes
        # ジョブごとの次工程が開始できる時刻の配列  ganttの中で出てきたそのジョブの最も遅い終了時間orリスケ地点が次工程の開始可能時刻
        self._next_process_starts = next_prosess_starts

    # job_numのジョブの次工程の機械番号を取得
    def get_machine(self, job_num):
        process_index = self._next_process_indexes[job_num]
        return self.jm_parent.get_machine(job_num, process_index)

    # job_numのジョブの次工程の処理時間を取得
    def get_process_time(self, job_num):
        process_index = self._next_process_indexes[job_num]
        return self.jm_parent.get_process_time(job_num, process_index)

    # job_numジョブの最も早い次工程の開始時刻を取得; この時間より後前開始できない
    def get_earliest(self, job_num):
        return self._next_process_starts[job_num]

    # job_numジョブを次工程に進める。次工程は現工程の終了時刻以降に開始できる
    def set_next_earliest(self, job_num, job_end):
        self._next_process_starts[job_num] = job_end
        self._next_process_indexes[job_num] += 1
