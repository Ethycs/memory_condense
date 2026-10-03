"""Freeze ten existing engineering tasks plus five new archived checkpoints."""
import argparse
import json
from pathlib import Path

from tools import engineering_research_battery as battery


def new_case(number, source, cutoff, anchor, title, module, task, requirements):
    case_id=f'E{number:02}'
    return dict(id=case_id,domain='engineering',split='validation',source=source,cutoff_turn=cutoff,
        request_anchor=anchor,title=title,task_kind='python_component',
        task=task+' Use only the Python standard library. Add executable unittest tests and design.md explaining assumptions, source requirements, and unverified claims. This is a bounded component prototype; do not claim to have run the archived experiment or implemented missing external repositories.',
        deliverables=[module+'.py','test_'+module+'.py','design.md'],
        criteria=[dict(id=f'{case_id}-C{i}',requirement=r,weight=1,supports=[dict(turn=t,quote=q)])
                  for i,(r,t,q) in enumerate(requirements,1)],
        public_checks=['Reject malformed inputs with ValueError or TypeError; preserve caller inputs.',
                       'Do not substitute speculative guarantees for an executable bounded prototype.'],
        structural_checks=['required_artifacts'],
        failure_classes=['missing_requirement','implementation_error','unsupported_claim','grader_error'])


def specification():
    original=json.loads(Path('evals/engineering_research/tasks.json').read_text(encoding='utf-8'))
    cases=[c for c in original['cases'] if c['domain']=='engineering']
    cases += [
        new_case(11,'Notes condensor/Layered_Context_Window_Graphs_beefa8c4_2025-07-13T03-26-26-136Z.txt',28,
            'segments text by rule','Lossless segmentation and graph reassembly','tape_graph',
            'Implement tape_graph.py. segment(text, boundaries) takes strictly increasing integer character offsets including 0 and len(text), returning records {id: "n0" etc., start, end, text} for adjacent intervals. Empty text with [0] returns []. to_graph(segments) returns {nodes: the records, edges: [[previous_id,next_id], ...]} preserving tape order. reassemble(segments, order) concatenates exact segment text in an explicitly supplied permutation of all IDs. Reject missing/repeated/unknown IDs, invalid offsets, and inconsistent segment records. The supplied offsets are rule k and the permutation is rule g; semantic rule discovery remains a pluggable concern.',
            [('Preserve original tape text and positions without losing content.',20,"cut up a piece of tape into nodes"),
             ('Represent ordered text pieces as an explicit graph.',10,'1 context window -> knowlege graph'),
             ('Reassemble notes using a separate explicit ordering rule.',22,'once we have the knowledge graph, then we can reassemble the notes right?'),
             ('Keep segmentation and reordering independent; attention-head access is not required by the prototype.',28,'segments text by rule  k and reorganizes by rule g')]),
        new_case(12,'NN Experimentation/Compaction__Neural_Network_Layer_Architecture_Strategies_326c9947_2025-07-10T18-51-28-423Z.txt',52,
            'each time a layer is added','Exact structural-zero layer compaction','zero_compaction',
            'Implement compact_layer(incoming, outgoing) in zero_compaction.py. Incoming is a nonempty rectangular h-by-n numeric list matrix; outgoing is a nonempty rectangular m-by-h matrix, with n,h,m positive. For this explicitly linear, zero-bias prototype, remove only hidden neurons whose entire incoming row is exactly zero. Return {incoming: retained rows, outgoing: retained columns, kept: original hidden indices, removed: original hidden indices}. Preserve order and all remaining weights, including tiny nonzero weights. All-zero incoming returns [] plus m empty outgoing rows. Reject mismatched/ragged/nonfinite matrices. Document why outgoing @ incoming @ x is preserved for every x under these assumptions, and why sample-zero activations, biased/nonlinear networks, and skip connections need separate treatment.',
            [('Discard structurally zero neurons while keeping nonzero contributions.',42,'we choose to discard some zeros'),
             ('Preserve surviving layer values rather than adding a new sparsity rule.',50,'compactify the layers not doing anything to the networks sparsity'),
             ('Expose a repeatable local-layer operation and original-index mapping.',52,'each time a layer is added'),
             ('State limitations when extending the linear prototype to skips and multilayer composition.',18,'skip {3 layers } skip')]),
        new_case(13,'NN Experimentation/Multi_Scale_Network_Architecture_Tuning_c621db40_2025-07-10T01-38-17-357Z.txt',44,
            'learning rate on the patched in network to half','Transferred-seed parameter planning','growth_plan',
            'Implement initialize_growth(seed_weights, expanded_ids, base_lr) in growth_plan.py. seed_weights maps nonempty string parameter IDs to finite numbers. expanded_ids is an ordered list of unique nonempty string IDs containing every seed ID. Return {parameters: [{id, value, lr, origin}, ...]} in expanded order. Preserve transferred seed values and assign them base_lr/2 with origin "seed"; new IDs get zero initialization, base_lr, and origin "new". Require finite positive base_lr and finite seed values. This explicit policy resolves the scope of the proposed half-rate experiment for the prototype. Include tests and a proposed comparison against same-rate transfer and an unseeded baseline in design.md; do not fabricate accuracy or an optimality guarantee.',
            [('Preserve the earlier seed when preparing a larger growth stage.',28,'use that we should the2% model for the seed for the next layer of complexity'),
             ('Apply the proposed half learning rate to transferred seed parameters.',44,'learning rate on the patched in network to half'),
             ('Account separately for transferred and newly allocated parameters during growth.',32,'growth should also be between course and med when they are used as a seed'),
             ('Treat selection of an optimal seed as an unverified claim to evaluate, not an established guarantee.',42,'so this is the optimum seed mnist right?')]),
        new_case(14,'Comb/AlgoSelect__Automated_Algorithm_Selection_Framework_867f26f1_2025-07-11T05-01-20-202Z.txt',34,
            'tune the dial','Bounded scalar interpolation tuner','mix_tuner',
            'Implement select_mix(left, right, score, alphas) in mix_tuner.py. left/right are equal nonempty finite-number vectors; alphas is a nonempty finite-number list in [0,1]. Evaluate each distinct alpha once in ascending order: value[i]=(1-alpha)*left[i]+alpha*right[i]. Call the supplied score(value) exactly once per distinct candidate and minimize its finite scalar return. Ties choose the lower alpha. Return {alpha, value, score, evaluations: [{alpha, score}, ...]}. Reject invalid input and nonfinite scores. Endpoints must work. Document candidate-grid limits, how a previous winning alpha can seed a later supplied grid, and why this neither guarantees a global optimum nor reimplements the unavailable repository.',
            [('Expose interpolation between two candidates as the explicit scalar control.',14,"interpolation between two algorithms with learning"),
             ('Keep candidate seeding explicit and testable.',18,'Weakest thing to me is the seeding'),
             ('Use measured callback feedback to choose among the supplied candidates without fabricated outcomes.',20,'What about autolearning'),
             ('Keep the control a single bounded dial and state the grid-search scope.',34,'only need to tune the dial')]),
        new_case(15,'Notes condensor/tape_to_graph_to_tape_conversion.txt',29,
            'masked out the non-activated sections','Neighborhood extraction with boundary provenance','roi_graph',
            'Implement extract_neighborhood(nodes, edges, seeds, hops) in roi_graph.py. Nodes is an ordered unique list of nonempty string IDs, edges is a list of directed [u,v] pairs using known IDs, seeds is a nonempty list of known IDs, and hops is a nonnegative integer. Select all nodes within hops using undirected adjacency for neighborhood membership. Return {nodes: selected IDs in original order, edges: original directed edges with both endpoints selected, boundary_edges: original directed edges with exactly one endpoint selected}. Preserve edge order and do not mutate inputs. Document that this preserves graph membership/provenance only: it does not prove activation/function preservation, catastrophe classification, cross-model correspondence, or a chain-map property.',
            [('Start with selected activation seeds and inspect their local neighborhoods.',5,'look at the neighborhood of these activations'),
             ('Preserve selected neighborhoods and record crossing connections instead of silently discarding their provenance.',19,'preserve the neighborhoods'),
             ('Deliver a concrete masked subgraph operation while distinguishing it from a proved compaction of a neural network.',29,'Reducing the network to just the neighborhoods'),
             ('Do not assume that a smaller model identifies equivalent neighborhoods in a larger model.',21,'identify the equivalent neighborhood in the larger model')]),
    ]
    protocol=dict(original['protocol'],unit='15 engineering checkpoints from ten session families; related checkpoints are correlated',
        first_pilot='E01 and E11, then remaining cases with operational stop gates',
        memory_lifecycle='Continuous ChatIO capture, recall on each actor action, inline exchange summaries, background ingest and learning, separate-process reopen',
        within_task_policy='Every new user/action/tool event is captured; full context retains the complete history and all current work; memory gets recalled evidence and the same bounded recent work window.',
        public_web='Both arms have identical bounded public search/open actions',
        evaluation_scope='15 component/design tasks adapted from real engineering sessions; natural prefixes, separate from 1M QA stress',
        cost_budget='At most 720 actor calls and 30 paired judge calls; source compilation local; no automatic retries')
    return dict(schema='engineering-research-task-spec-v1',protocol=protocol,cases=cases)


def build(output):
    spec=output/'tasks.json'
    battery.publish(spec,specification())
    result=battery.build(spec,battery.DEFAULT_SOURCE,output/'bundle')
    print(json.dumps(result),flush=True)
    print(json.dumps(battery.audit(output/'bundle')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output.resolve())
