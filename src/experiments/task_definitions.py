"""
Task Definitions for Multi-Task Procedural Assistance RL Simulation

This module defines 7 procedural tasks from the PrISM dataset, ranging from
simple 8-step tasks (MakeCereal) to complex 20-step tasks (latte_making).

Each task is defined with:
- Step-by-step procedure with descriptions
- Per-step criticality values (multipliers for failure costs)
- Base failure and interruption costs
- Domain classification (cooking, technical, crafting)

Tasks included:
1. latte_making (20 steps) - Technical, machine-based coffee preparation
2. MakeCereal (8 steps) - Simple breakfast preparation
3. MakeCoffee (8 steps) - Pod-based coffee brewing
4. MakeSandwich (9 steps) - Sandwich assembly with flexible prep phase
5. MakeStencil (17 steps) - Laser cutting and painting (safety-critical)
6. MakeTea (9 steps) - Kettle-based tea brewing
7. cooking (14 steps) - Sequential cooking with stove operation
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class StepDefinition:
    """Defines a single step in a procedural task.

    Attributes:
        name: Unique identifier for the step (e.g., "get_onion")
        description: Human-readable description of what happens in this step
        criticality: Multiplier for failure cost (1.0 = normal, 2.0 = twice as costly to fail)
        mean_duration: Expected duration in seconds (default: 30)
        std_duration: Standard deviation of duration (default: 10)
    """
    name: str
    description: str
    criticality: float = 1.0
    mean_duration: int = 30
    std_duration: int = 10


@dataclass
class TaskDefinition:
    """Defines a complete procedural task with all steps and cost parameters.

    Attributes:
        task_name: Unique identifier for the task
        steps: List of step definitions in execution order
        base_failure_cost: Base cost for a procedural failure
        interruption_cost: Cost of a single assistant interruption
        domain: Task domain (cooking, technical, crafting)
    """
    task_name: str
    steps: List[StepDefinition]
    base_failure_cost: float = 20.0
    interruption_cost: float = 5.0
    domain: str = "unknown"

    @property
    def n_steps(self) -> int:
        """Returns the number of steps in this task."""
        return len(self.steps)

    @property
    def step_names(self) -> List[str]:
        """Returns list of step names in order."""
        return [step.name for step in self.steps]

    def get_step_failure_cost(self, step_index: int) -> float:
        """Calculate failure cost for a specific step.

        Args:
            step_index: Index of the step (0-based)

        Returns:
            Failure cost = base_failure_cost * step_criticality
        """
        if 0 <= step_index < len(self.steps):
            return self.base_failure_cost * self.steps[step_index].criticality
        return self.base_failure_cost

    def get_step_criticality(self, step_index: int) -> float:
        """Get criticality value for a specific step.

        Args:
            step_index: Index of the step (0-based)

        Returns:
            Criticality multiplier (1.0 = normal, >1.0 = critical)
        """
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index].criticality
        return 1.0


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

def create_latte_making_task() -> TaskDefinition:
    """20-step latte making task (most complex, machine-based).

    Domain: Technical (espresso machine operation)
    Characteristics:
    - Linear workflow with machine constraints
    - Critical steps: brew_coffee, steam_milk (thermal/pressure risks)
    - High base costs due to equipment and ingredient waste
    """
    steps = [
        StepDefinition("gather_ingredients", "Gather milk, coffee beans, cup", criticality=1.0),
        StepDefinition("grind_beans", "Grind coffee beans to espresso consistency", criticality=1.2),
        StepDefinition("prepare_portafilter", "Fill and tamp portafilter", criticality=1.3),
        StepDefinition("attach_portafilter", "Attach portafilter to espresso machine", criticality=1.4),
        StepDefinition("place_cup", "Place cup under portafilter spout", criticality=1.1),
        StepDefinition("brew_coffee", "Extract espresso shot (25-30 sec)", criticality=2.0),
        StepDefinition("remove_portafilter", "Remove and knock out spent grounds", criticality=1.1),
        StepDefinition("pour_milk", "Pour cold milk into pitcher", criticality=1.0),
        StepDefinition("position_pitcher", "Position pitcher under steam wand", criticality=1.2),
        StepDefinition("steam_milk", "Steam milk to 150-155°F", criticality=2.0),
        StepDefinition("texture_milk", "Create microfoam texture", criticality=1.5),
        StepDefinition("tap_pitcher", "Tap pitcher to remove large bubbles", criticality=1.0),
        StepDefinition("swirl_milk", "Swirl milk to maintain texture", criticality=1.2),
        StepDefinition("pour_milk_slowly", "Begin pouring milk slowly into center", criticality=1.4),
        StepDefinition("create_pattern", "Pour latte art pattern", criticality=1.3),
        StepDefinition("finish_pour", "Complete milk pour", criticality=1.2),
        StepDefinition("wipe_cup", "Wipe any drips from cup exterior", criticality=1.0),
        StepDefinition("clean_wand", "Purge and wipe steam wand", criticality=1.5),
        StepDefinition("clean_portafilter", "Rinse portafilter basket", criticality=1.1),
        StepDefinition("serve_latte", "Present finished latte", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="latte_making",
        steps=steps,
        base_failure_cost=25.0,  # High due to equipment and ingredient waste
        interruption_cost=5.0,
        domain="technical"
    )


def create_make_cereal_task() -> TaskDefinition:
    """8-step cereal making task (simplest task).

    Domain: Cooking (breakfast preparation)
    Characteristics:
    - Short linear workflow
    - Minimal criticality variation
    - Low stakes (cheap ingredients, easy recovery)
    - Steps 2-3 can swap order
    """
    steps = [
        StepDefinition("get_bowl", "Retrieve bowl from cabinet", criticality=1.0),
        StepDefinition("get_cereal", "Get cereal box from pantry", criticality=1.0),
        StepDefinition("get_milk", "Get milk from refrigerator", criticality=1.0),
        StepDefinition("pour_cereal", "Pour cereal into bowl", criticality=1.2),
        StepDefinition("pour_milk", "Pour milk over cereal", criticality=1.3),
        StepDefinition("get_spoon", "Get spoon from drawer", criticality=1.0),
        StepDefinition("return_items", "Return cereal and milk to storage", criticality=1.0),
        StepDefinition("serve_cereal", "Present cereal bowl", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="make_cereal",
        steps=steps,
        base_failure_cost=10.0,  # Low stakes
        interruption_cost=5.0,
        domain="cooking"
    )


def create_make_coffee_task() -> TaskDefinition:
    """8-step pod-based coffee making task.

    Domain: Cooking (beverage preparation)
    Characteristics:
    - Simple machine operation (pod-based)
    - Steps 2-3 frequently swap in real data
    - Critical step: brew_coffee (machine operation)
    """
    steps = [
        StepDefinition("get_mug", "Retrieve coffee mug", criticality=1.0),
        StepDefinition("get_coffee_pod", "Get coffee pod", criticality=1.0),
        StepDefinition("add_water", "Fill water reservoir", criticality=1.2),
        StepDefinition("insert_pod", "Insert pod into machine", criticality=1.3),
        StepDefinition("place_mug", "Place mug under dispenser", criticality=1.2),
        StepDefinition("brew_coffee", "Start brew cycle", criticality=1.8),
        StepDefinition("remove_pod", "Remove and discard used pod", criticality=1.1),
        StepDefinition("serve_coffee", "Present finished coffee", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="make_coffee",
        steps=steps,
        base_failure_cost=12.0,
        interruption_cost=5.0,
        domain="cooking"
    )


def create_make_sandwich_task() -> TaskDefinition:
    """9-step sandwich making task with flexible prep phase.

    Domain: Cooking (meal preparation)
    Characteristics:
    - DAG-like structure: prep phase (steps 2-5) highly flexible
    - Execution phase (steps 6-9) more constrained
    - Critical step: prepare_sandwich (assembly)
    """
    steps = [
        StepDefinition("get_ingredients", "Gather bread, fillings, condiments", criticality=1.0),
        StepDefinition("get_plate", "Get plate for sandwich", criticality=1.0),
        StepDefinition("get_knife", "Get knife for spreading", criticality=1.0),
        StepDefinition("arrange_workspace", "Arrange ingredients on counter", criticality=1.0),
        StepDefinition("prepare_bread", "Lay out bread slices", criticality=1.1),
        StepDefinition("spread_condiments", "Spread mayo/mustard on bread", criticality=1.2),
        StepDefinition("add_fillings", "Add meat, cheese, vegetables", criticality=1.3),
        StepDefinition("prepare_sandwich", "Assemble and cut sandwich", criticality=1.5),
        StepDefinition("serve_sandwich", "Place sandwich on plate", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="make_sandwich",
        steps=steps,
        base_failure_cost=15.0,
        interruption_cost=5.0,
        domain="cooking"
    )


def create_make_stencil_task() -> TaskDefinition:
    """17-step stencil making task (safety-critical laser cutting).

    Domain: Crafting (laser cutter operation)
    Characteristics:
    - Two phases: laser cutting (safety-critical) + painting
    - Highest criticality: exhaust system checks (fire/fume hazard)
    - Critical: laser operation steps (equipment damage, injury risk)
    - Highest base costs due to safety and equipment concerns
    """
    steps = [
        StepDefinition("design_stencil", "Create or select stencil design", criticality=1.0),
        StepDefinition("prepare_file", "Prepare vector file for laser", criticality=1.2),
        StepDefinition("check_exhaust", "Verify exhaust system running", criticality=2.5),
        StepDefinition("select_material", "Choose appropriate material (cardstock/mylar)", criticality=1.3),
        StepDefinition("load_material", "Load material into laser bed", criticality=1.4),
        StepDefinition("focus_laser", "Focus laser to material thickness", criticality=1.8),
        StepDefinition("set_parameters", "Set power, speed, frequency", criticality=1.6),
        StepDefinition("position_design", "Position design on material", criticality=1.2),
        StepDefinition("start_cutting", "Start laser cutting process", criticality=2.0),
        StepDefinition("monitor_cutting", "Monitor for flare-ups or issues", criticality=2.2),
        StepDefinition("remove_cutout", "Remove finished stencil from bed", criticality=1.3),
        StepDefinition("clean_edges", "Remove any burrs or rough edges", criticality=1.2),
        StepDefinition("prepare_surface", "Clean surface to be painted", criticality=1.1),
        StepDefinition("position_stencil", "Position stencil on surface", criticality=1.3),
        StepDefinition("secure_stencil", "Tape or hold stencil firmly", criticality=1.4),
        StepDefinition("apply_paint", "Apply paint through stencil", criticality=1.5),
        StepDefinition("remove_stencil", "Carefully remove stencil", criticality=1.2),
    ]

    return TaskDefinition(
        task_name="make_stencil",
        steps=steps,
        base_failure_cost=30.0,  # Highest due to safety and equipment risks
        interruption_cost=5.0,
        domain="crafting"
    )


def create_make_tea_task() -> TaskDefinition:
    """9-step tea making task.

    Domain: Cooking (beverage preparation)
    Characteristics:
    - Simple kettle-based workflow
    - Steps 3-4 can swap order
    - Critical step: pour_tea (scalding risk)
    """
    steps = [
        StepDefinition("get_mug", "Retrieve tea mug", criticality=1.0),
        StepDefinition("get_tea_bag", "Get tea bag or loose leaf", criticality=1.0),
        StepDefinition("fill_kettle", "Fill kettle with water", criticality=1.1),
        StepDefinition("heat_water", "Boil water in kettle", criticality=1.5),
        StepDefinition("place_tea_bag", "Place tea bag in mug", criticality=1.0),
        StepDefinition("pour_water", "Pour hot water over tea", criticality=1.8),
        StepDefinition("steep_tea", "Let tea steep (3-5 minutes)", criticality=1.2),
        StepDefinition("remove_tea_bag", "Remove and discard tea bag", criticality=1.1),
        StepDefinition("serve_tea", "Present finished tea", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="make_tea",
        steps=steps,
        base_failure_cost=12.0,
        interruption_cost=5.0,
        domain="cooking"
    )


def create_cooking_task() -> TaskDefinition:
    """14-step sequential cooking task (stove-based).

    Domain: Cooking (stovetop meal preparation)
    Characteristics:
    - Sequential cooking workflow with timing constraints
    - Critical steps: stove operation (burn risk, food safety)
    - Time-sensitive (overcooking/undercooking risks)
    """
    steps = [
        StepDefinition("gather_ingredients", "Gather all cooking ingredients", criticality=1.0),
        StepDefinition("get_cookware", "Get pots, pans, utensils", criticality=1.0),
        StepDefinition("prep_ingredients", "Wash, chop, measure ingredients", criticality=1.1),
        StepDefinition("preheat_pan", "Preheat pan on stove", criticality=1.5),
        StepDefinition("add_oil", "Add cooking oil to pan", criticality=1.3),
        StepDefinition("add_first_ingredients", "Add first ingredients to pan", criticality=1.4),
        StepDefinition("saute_ingredients", "Sauté/cook first ingredients", criticality=1.8),
        StepDefinition("add_seasonings", "Add spices and seasonings", criticality=1.2),
        StepDefinition("add_remaining", "Add remaining ingredients", criticality=1.5),
        StepDefinition("cook_thoroughly", "Cook until done (check temperature)", criticality=2.0),
        StepDefinition("adjust_seasoning", "Taste and adjust seasoning", criticality=1.2),
        StepDefinition("turn_off_stove", "Turn off burner", criticality=1.6),
        StepDefinition("plate_food", "Transfer food to serving plate", criticality=1.3),
        StepDefinition("serve_meal", "Present finished meal", criticality=1.0),
    ]

    return TaskDefinition(
        task_name="cooking",
        steps=steps,
        base_failure_cost=20.0,  # High due to food safety and burn risks
        interruption_cost=5.0,
        domain="cooking"
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def load_task_definitions() -> Dict[str, TaskDefinition]:
    """Load all 7 task definitions.

    Returns:
        Dictionary mapping task names to TaskDefinition objects.

    Example:
        tasks = load_task_definitions()
        cereal_task = tasks['make_cereal']
        print(f"Task has {cereal_task.n_steps} steps")
    """
    return {
        "latte_making": create_latte_making_task(),
        "make_cereal": create_make_cereal_task(),
        "make_coffee": create_make_coffee_task(),
        "make_sandwich": create_make_sandwich_task(),
        "make_stencil": create_make_stencil_task(),
        "make_tea": create_make_tea_task(),
        "cooking": create_cooking_task(),
    }


def get_task_definition(task_name: str) -> TaskDefinition:
    """Get a single task definition by name.

    Args:
        task_name: Name of the task (e.g., "make_cereal")

    Returns:
        TaskDefinition object for the specified task.

    Raises:
        KeyError: If task_name is not recognized.
    """
    tasks = load_task_definitions()
    if task_name not in tasks:
        available = ', '.join(tasks.keys())
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return tasks[task_name]


def print_task_summary():
    """Print summary table of all tasks."""
    tasks = load_task_definitions()

    print("\n" + "="*80)
    print("TASK DEFINITIONS SUMMARY")
    print("="*80)
    print(f"{'Task':<20} {'Steps':<8} {'Domain':<12} {'Fail Cost':<12} {'Int Cost':<10}")
    print("-"*80)

    for task_name, task_def in sorted(tasks.items(), key=lambda x: x[1].n_steps):
        print(f"{task_name:<20} {task_def.n_steps:<8} {task_def.domain:<12} "
              f"{task_def.base_failure_cost:<12.1f} {task_def.interruption_cost:<10.1f}")

    print("-"*80)
    print(f"Total tasks: {len(tasks)}")
    print(f"Step count range: {min(t.n_steps for t in tasks.values())} - "
          f"{max(t.n_steps for t in tasks.values())}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print summary when run directly
    print_task_summary()

    # Show detailed info for one task
    print("\nDETAILED EXAMPLE: make_cereal")
    print("-"*80)
    cereal = get_task_definition("make_cereal")
    print(f"Task: {cereal.task_name}")
    print(f"Domain: {cereal.domain}")
    print(f"Steps: {cereal.n_steps}")
    print(f"Base failure cost: {cereal.base_failure_cost}")
    print(f"Interruption cost: {cereal.interruption_cost}")
    print("\nSteps:")
    for i, step in enumerate(cereal.steps):
        failure_cost = cereal.get_step_failure_cost(i)
        print(f"  {i+1}. {step.name:<20} (criticality={step.criticality:.1f}, "
              f"fail_cost={failure_cost:.1f})")
        print(f"     {step.description}")
