from PySide6.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem, QVBoxLayout
from numba.cuda.types import grid_group
from casys.viz.tools.core import ToolPlugin, ToolEvent, ToolContext

class InspectTool(ToolPlugin):
    name = 'Inspect tool'

    def on_event(self, ev: ToolEvent, ctx: ToolContext) -> None:
        if ev.kind != 'up': return

        x, y = ev.gpos

        sim_mgr = ctx.sim_mgr
        window = ctx.window

        x %= sim_mgr.dims[0]
        y %= sim_mgr.dims[1]

        world_schema = sim_mgr.sim.world_schema
        t = sim_mgr.sim.timestamp
        state = sim_mgr.get_current_state()

        # Build schema and state mapping
        dlg = QDialog(ctx.window)
        dlg.setWindowTitle(f'Inspect {x},{y} | T = {t}')
        dlg.resize(300, 400)
        tree = QTreeWidget(dlg)
        tree.setHeaderLabels(['Field', 'Value'])
        for group in world_schema.get_children():
            group_name = group.name
            group_item = QTreeWidgetItem(tree, [group_name])
            for field in group.get_children():
                field_name = field.name
                key = field.canonical_name()
                val = state[key][x, y]
                if key in window.ui_model.inspect_processors:
                    value_str = window.ui_model.inspect_processors[key](val)
                else:
                    value_str = str(val)
                QTreeWidgetItem(group_item, [field_name, value_str])

        layout = QVBoxLayout(dlg)
        layout.addWidget(tree)
        dlg.setModal(False)
        dlg.show()